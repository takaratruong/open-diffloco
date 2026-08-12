# G1 RMR Action Noise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace uniform scalar SHAC action noise with the pinned learned RMR per-joint vector, render a matched noisy/deterministic control pair, and run one bounded causal continuation.

**Architecture:** Extend only the SHAC noise schedule boundary to accept validated scalar-or-vector values while preserving the existing pathwise rollout. Add a small provenance module for the pinned RMR joint names/std values, a matched evaluator layered over the existing G1 evaluator, and an immutable continuation runner modeled on the zero-bootstrap runner.

**Tech Stack:** Python, JAX, NumPy, MuJoCo/MJX, pytest, Ruff.

## Global Constraints

- The vector must be the exact float32 `std` tensor from RMR checkpoint SHA-256 `5174a0f1dc8c83ef9ea45769c3b0f19383e5aeeafea2171433f8e7bb88b21746`.
- Vector order must exactly equal `env.actor_joint_names`; mismatch fails closed.
- Existing scalar-noise runs must retain identical shapes, values, and gradients.
- The matched pair must share checkpoint, reset state, assets, solver, and seed.
- The continuation changes only action noise and its explicit resume authority.
- Training reward and noisy rollout behavior cannot select a checkpoint.

---

### Task 1: Scalar-or-vector SHAC noise boundary

**Files:**
- Create: `src/core/rmr_action_noise.py`
- Modify: `src/algorithms/shac/algorithm.py`
- Test: `tests/test_shac_action_noise.py`

**Interfaces:**
- Produces: `RMR_ACTION_STD`, `RMR_ACTION_STD_JOINT_NAMES`, `validate_action_noise_std(value, *, action_dim, actor_joint_names) -> jax.Array`, and `resolve_action_noise_resume_settings(...)`.

- [ ] **Step 1: Write failing tests** for exact pinned values/order, scalar compatibility, vector broadcast over `(H, 29)` epsilon, invalid rank/shape/nonfinite/negative values, joint-order mismatch, and unauthorized versus authorized resume changes.
- [ ] **Step 2: Run RED:** `pytest -q tests/test_shac_action_noise.py`; expect missing module/interfaces.
- [ ] **Step 3: Implement minimal validation and schedule wiring.** Preserve scalar JAX arrays as scalars; represent vector hparams as JSON lists; linearly interpolate elementwise; multiply the rollout epsilon by the resulting scalar or `(29,)` vector.
- [ ] **Step 4: Run GREEN plus neighbors:** `pytest -q tests/test_shac_action_noise.py tests/test_shac_gradient_aggregation.py tests/test_shac_microbatch.py`.
- [ ] **Step 5: Commit only Task 1 files.**

### Task 2: Provenance-bound matched rollout pair

**Files:**
- Create: `tools/evaluate_g1_action_noise_pair.py`
- Test: `tests/test_g1_action_noise_pair.py`

**Interfaces:**
- Consumes: selected E-20260812-008 checkpoint, existing `evaluate_g1_tracking` rollout primitives, `RMR_ACTION_STD`.
- Produces: `deterministic/` and `rmr-noisy/` JSON/NPZ/MP4/contact-sheet artifacts plus `action_noise_pair.json`.

- [ ] **Step 1: Write failing tests** for exact shared-reset identity, deterministic zero noise, seeded RMR samples, per-step joint order, provenance hashes, artifact completeness, and fail-closed pair aggregation.
- [ ] **Step 2: Run RED:** `pytest -q tests/test_g1_action_noise_pair.py`; expect missing tool.
- [ ] **Step 3: Implement the pair evaluator** by injecting caller-owned action perturbations immediately before `env.step`; do not change the policy or environment evaluator semantics.
- [ ] **Step 4: Run GREEN and one CPU/no-render real-checkpoint smoke.**
- [ ] **Step 5: Commit only Task 2 files.**

### Task 3: Immutable 32-update RMR-noise continuation

**Files:**
- Create: `tools/run_g1_rmr_action_noise_continuation.py`
- Test: `tests/test_g1_rmr_action_noise_continuation.py`

**Interfaces:**
- Consumes: exact E-20260812-008 selected checkpoint/hparams and Task 1 vector resume authority.
- Produces: preflight, four dense checkpoints, training validation, and deterministic five-phase selection inputs.

- [ ] **Step 1: Write failing tests** proving the only inherited-kwargs deltas are endpoint, vector noise start/end, schedule endpoint, and explicit noise-resume authority; require exact vector/order/hash, 32 updates, zero assistance/bootstrap, frozen parent, finite five-bin CAGrad, and exact checkpoint cadence.
- [ ] **Step 2: Run RED:** `pytest -q tests/test_g1_rmr_action_noise_continuation.py`; expect missing runner.
- [ ] **Step 3: Implement minimal immutable runner** from `run_g1_zero_bootstrap_continuation.py`, without scientific CLI overrides.
- [ ] **Step 4: Run GREEN, Ruff, pycompile, focused SHAC tests, and `git diff --check`.**
- [ ] **Step 5: Commit and request bounded code/scientific-contract review.**

### Task 4: Register, execute, evaluate, and publish

**Files:**
- Create in lab: `research/experiments/E-20260812-009.yaml`
- Modify after evidence: `research/state/current.yaml`, experiment result, qualifying finding, generated vault Dashboard/experiment page, and `runtime/vault-previews.yaml`.

**Interfaces:**
- Consumes: reviewed clean code commits and exact pinned artifacts.
- Produces: canonical matched-rollout evidence, deterministic five-phase checkpoint grids, selected rollout media, and one registered verdict.

- [ ] **Step 1: Register, validate, dry-run, and approve** the exact one-change experiment.
- [ ] **Step 2: Run the matched current-policy rollout pair** and publish both videos/contact sheets to the Dashboard preview tree.
- [ ] **Step 3: Execute the 32-update continuation** through `python -m tools.runexp E-20260812-009`.
- [ ] **Step 4: Deterministically evaluate all four checkpoints** at five phases and apply the preregistered minimum/median/mean/earliest selector.
- [ ] **Step 5: Render selected deterministic and noisy rollouts, curate the verdict, rebuild the vault, verify all artifacts, and commit only owned files.**
