# G1 Pseudo-Huber Velocity Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, provenance-bound pseudo-Huber treatment for the G1 body linear- and angular-velocity tracking rewards while preserving every legacy checkpoint's exponential objective by default.

**Architecture:** Keep the six existing reward components and all component weights unchanged. Add one strict `tracking_velocity_kernel` setting with values `exponential` and `pseudo_huber`; only the two velocity components consult it. Thread the setting through the G1 environment, SHAC hparams/resume boundary, training builder, and evaluator so training and evidence reconstruction cannot silently disagree.

**Tech Stack:** Python, JAX, MJX, unittest/pytest, Ruff.

## Global Constraints

- The legacy default is exactly `exponential`.
- The pseudo-Huber component is `2 - sqrt(1 + 2 * normalized_mean_squared_error)`, giving value one and the same first derivative with respect to normalized MSE as `exp(-x)` at zero while retaining a non-vanishing residual gradient at large error.
- Position and orientation rewards, reward weights, regularizers, action boundary, solver, observation, reset distribution, CAGrad, horizon, and batch size remain unchanged.
- Missing checkpoint metadata means the legacy exponential kernel; changing the kernel on resume requires an explicit authority flag.
- Invalid kernel names or non-boolean authority values fail closed before environment compilation.

---

### Task 1: Reward and environment contract

**Files:**
- Modify: `src/envs/g1_tracking/reward.py`
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `tests/test_g1_tracking_reward.py`
- Modify: `tests/test_g1_tracking_environment.py`

**Interfaces:**
- Consumes: existing normalized body linear/angular velocity MSEs.
- Produces: `tracking_velocity_kernel: str` on `G1TrackingEnv` and `rmr_tracking_reward(..., velocity_kernel: str = "exponential")`.

- [ ] **Step 1: Write failing reward tests**

Add tests proving the legacy exponential values are unchanged, the pseudo-Huber value is exact at a known normalized MSE, its large-error residual gradient is finite/nonzero and points toward the reference, and invalid names fail closed.

- [ ] **Step 2: Run reward tests and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_tracking_reward.py`

Expected: failures because `velocity_kernel` is not accepted.

- [ ] **Step 3: Implement the minimal reward kernel**

Add a private strict kernel dispatcher. Preserve the existing exponential branch byte-for-byte and use `2.0 - jp.sqrt(1.0 + 2.0 * normalized_mse)` only for pseudo-Huber velocity components.

- [ ] **Step 4: Write and run the environment RED/GREEN tests**

Test default legacy behavior, explicit pseudo-Huber forwarding, and invalid constructor values. Add the validated setting to `G1TrackingEnv` and forward it from `_tracking_reward_from_body_state`.

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_tracking_reward.py tests/test_g1_tracking_environment.py -k 'reward or velocity_kernel'`

Expected: all selected tests pass.

### Task 2: Training, resume, and evaluation provenance

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tools/run_g1_tracking_rmr50_shac.py`
- Modify: `tools/evaluate_g1_tracking.py`
- Modify: `tests/test_shac_exact_resume.py`
- Modify: `tests/test_g1_tracking_evaluator.py`
- Modify: the closest focused runner-builder test.

**Interfaces:**
- Consumes: `tracking_velocity_kernel` and `allow_resume_tracking_velocity_kernel_change`.
- Produces: exact env construction, hparams persistence, legacy restoration, strict resume authority, and evaluator reconstruction.

- [ ] **Step 1: Write failing provenance tests**

Cover fresh pseudo-Huber forwarding, absent hparams restoring exponential, exact resume matching, unauthorized mismatch rejection, authorized mismatch acceptance, malformed metadata rejection, and training-distribution evaluator restoration.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_exact_resume.py tests/test_g1_tracking_evaluator.py -k 'velocity_kernel'`

Expected: failures because the resolver and plumbing do not exist.

- [ ] **Step 3: Implement minimal plumbing**

Add a pure resume resolver, train arguments and validation, G1 environment kwargs, hparams fields, builder arguments, evaluator resolver/constructor/CLI forwarding, and report metadata. Do not change any default.

- [ ] **Step 4: Run focused GREEN and neighboring regression tests**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_exact_resume.py tests/test_g1_tracking_reward.py tests/test_g1_tracking_environment.py tests/test_g1_tracking_evaluator.py tools/run_g1_tracking_rmr50_shac.py`

Expected: all tests pass (the tool path may collect zero tests without error).

### Task 3: Scientific discriminator and launch boundary

**Files:**
- Create: `tools/run_g1_pseudo_huber_velocity_h24_walk.py`
- Create: `tests/test_g1_pseudo_huber_velocity_h24_walk.py`
- Create or update: registered experiment record in `diffsim2real-lab/research/experiments/`

**Interfaces:**
- Consumes: the exact retained E023 walk recipe and the new kernel setting.
- Produces: one treatment whose sole causal delta is `tracking_velocity_kernel=pseudo_huber`, plus early per-term gradient diagnostics and bounded H24 checkpoints.

- [ ] **Step 1: Write failing runner-contract tests**

Require exact E023 solver/action/noise/reset/observation/CAGrad/batch/horizon settings, kernel `pseudo_huber`, legacy root-position observation disabled, immutable checkpoint cadence, code/reference/model/controller provenance, and fail-closed artifact validation.

- [ ] **Step 2: Run the runner tests and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_pseudo_huber_velocity_h24_walk.py`

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement the smallest runner by reusing reviewed E023 gates**

Build E023 kwargs, change only the kernel, retain the existing phase-grid evaluator and diagnostic artifacts, and classify early evidence without promoting a policy from one stochastic run.

- [ ] **Step 4: Verify, register, and launch**

Run focused tests, Ruff on changed Python files, `python -m py_compile` on changed Python files, and `git diff --check`. Register the treatment with exact commit and asset hashes. Launch only after all no-GPU preflight gates pass.

## Self-Review

- Spec coverage: legacy compatibility, kernel math, environment wiring, resume authority, evaluator reconstruction, runner provenance, and bounded scientific evidence each have a task.
- Placeholder scan: no deferred implementation or unspecified validation remains.
- Type consistency: the setting is a strict string end-to-end; the authority flag is a strict boolean; all hparams use the same field names.
