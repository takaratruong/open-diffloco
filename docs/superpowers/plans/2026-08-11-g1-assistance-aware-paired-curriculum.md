# G1 Assistance-Aware Paired Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a causal paired test of scalar assistance observability in the E012 residual policy.

**Architecture:** Append an unnormalized scalar only at the trainable residual adapter boundary, migrate its first kernel and Adam moments with an exact-zero row, and compare true-scalar versus zero-scalar arms under one identical continuous assistance curriculum. Keep the frozen parent, observation history, normalizer, wrench controller, and zero-wrench evaluator unchanged.

**Tech Stack:** Python 3.10, JAX, Flax, Optax, MJX, pytest, Ruff, the DiffSim2Real experiment registry.

## Global Constraints

- Resume only E012 step 1,671,168 with SHA-256 `f375cadc9bf8b5cef26fc7414133071910fed393344c99bbacffea963aa9f4f7`.
- Expose only scalar assistance magnitude; never expose wrench or oracle errors.
- Both arms have the same 329-input adapter architecture and identical physical assistance samples.
- Evaluation and export use exact-zero assistance.
- Assisted reward never selects a checkpoint.
- Write tests before production code and observe each expected failure.

---

### Task 1: Continuous sampling and scalar adapter boundary

**Files:**
- Modify: `tests/test_shac_torso_wrench_curriculum.py`
- Modify: `tests/test_shac_residual_preview_adapter.py`
- Modify: `src/algorithms/shac/torso_wrench_curriculum.py`
- Modify: `src/algorithms/shac/residual_preview_adapter.py`

**Interfaces:**
- Produces: `sample_assistance_scales(..., continuous: bool)` and optional `assistance_scale` accepted by `apply_frozen_preview_residual`.

- [ ] **Step 1: Write failing sampler tests** proving deterministic `[0, cap]` samples, a statistical exact-zero cohort, bounded finite values, and exact all-zero output at cap zero.
- [ ] **Step 2: Run** `pytest -q tests/test_shac_torso_wrench_curriculum.py` and verify the new tests fail because continuous sampling is absent.
- [ ] **Step 3: Implement** continuous sampling by drawing the zero mask and magnitude from separate deterministic key splits, then using `where(mask, 0, magnitude * cap)`.
- [ ] **Step 4: Write failing adapter tests** proving true scalar changes a deliberately nonzero toy adapter, blind/omitted scalar means exact zero, and invalid scalar shapes/ranges fail closed.
- [ ] **Step 5: Run** `pytest -q tests/test_shac_residual_preview_adapter.py` and verify the new tests fail because conditioning is absent.
- [ ] **Step 6: Implement** scalar concatenation at the newest-frame residual boundary while leaving the parent actor input untouched.
- [ ] **Step 7: Run both focused modules** and require all tests to pass.
- [ ] **Step 8: Commit** only the four scoped files.

### Task 2: Exact checkpoint and optimizer migration

**Files:**
- Modify: `tests/test_shac_residual_preview_adapter.py`
- Modify: `src/algorithms/shac/residual_preview_adapter.py`
- Modify: `tests/test_shac_torso_wrench_curriculum.py`
- Modify: `src/algorithms/shac/algorithm.py`

**Interfaces:**
- Produces: `migrate_residual_adapter_assistance_conditioning(...)` and a JSON-safe migration report.
- Consumes: the optional scalar adapter boundary from Task 1.

- [ ] **Step 1: Write failing migration tests** that start from a trained nonzero adapter and Adam state, append exactly one zero kernel/moment row, preserve every pre-existing value and optimizer count, and reproduce the original action at scalar zero.
- [ ] **Step 2: Run the focused migration tests** and verify the missing migration API is the failure.
- [ ] **Step 3: Implement minimal migration and audit helpers** for the registered Adam residual checkpoint shape.
- [ ] **Step 4: Write failing train-contract tests** for default-off conditioning, explicit resume authority, aware versus blind scalar routing, saved hparams, and exact-zero evaluator fallback.
- [ ] **Step 5: Run the contract tests** and verify the new train flags/wiring are absent.
- [ ] **Step 6: Wire train** so both paired arms migrate to 329 inputs; true or zero scalar is routed per arm; telemetry records finite, bounded, truthful conditioning; expected trainable parameter count increases by exactly 256.
- [ ] **Step 7: Run focused SHAC, residual, assistance, microbatch, CAGrad, and evaluator tests** and require green.
- [ ] **Step 8: Commit** the scoped migration and integration files.

### Task 3: Immutable paired runners and preflight

**Files:**
- Create: `tools/run_g1_assistance_observability_pair.py`
- Create: `tests/test_g1_assistance_observability_pair_runner.py`
- Modify: `tools/evaluate_g1_flax_phase_grid.py` only if zero-conditioning inference is not already contained in the shared adapter API.

**Interfaces:**
- Produces: two immutable kwargs dictionaries whose only difference is `actor_observe_torso_wrench_assistance`.

- [ ] **Step 1: Write failing runner tests** asserting exact E012 parent, identical paired kwargs and seeds, continuous sampler enabled, 25% exact-zero cohort, fade and zero-tail endpoints, no scientific CLI overrides, and a one-field aware/blind delta.
- [ ] **Step 2: Run the runner tests** and verify the module is missing.
- [ ] **Step 3: Implement the paired runner** with fixed output subdirectories and concurrent child processes pinned to distinct GPUs when two devices are available.
- [ ] **Step 4: Add a no-GPU preflight** that validates parent SHA, code/model/controller/reference hashes, paired delta, migration action equivalence, and output isolation before either child starts.
- [ ] **Step 5: Run runner, evaluator, Ruff, py_compile, and diff checks** and require green.
- [ ] **Step 6: Commit** runner and tests.

### Task 4: Register, launch, evaluate, and curate

**Files:**
- Create in lab repository: paired experiment registry YAML and deterministic evaluator/selector only where existing reusable tooling is insufficient.
- Modify in lab repository after evidence: `research/state/current.yaml` and generated vault projections through the repository tools.

**Interfaces:**
- Consumes: immutable paired runner and exact E012 checkpoint.
- Produces: separate arm run directories, checkpoint audits, five-phase zero-wrench artifacts, comparison JSON, and curated decision.

- [ ] **Step 1: Register one approved paired experiment** with exactly one arm-to-arm causal change, hashes, budget, outcome map, and zero-wrench selector.
- [ ] **Step 2: Validate the registry and command contract** without GPU execution.
- [ ] **Step 3: Launch both arms concurrently** and monitor compilation plus first-update scalar, wrench, CAGrad, finite-gradient, and migration telemetry.
- [ ] **Step 4: Evaluate every eligible zero-tail checkpoint** on the five-phase grid, producing metrics, trajectories, videos, and montages.
- [ ] **Step 5: Select within each arm by min/median/mean survival**, then compare aware, blind, and E012.
- [ ] **Step 6: Curate the supported conclusion**, verify research consistency, commit, and push code and lab repositories.
