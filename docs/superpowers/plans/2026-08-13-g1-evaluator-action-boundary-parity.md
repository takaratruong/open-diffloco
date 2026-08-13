# G1 Evaluator Action-Boundary Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make G1 evaluation apply the exact SHAC post-noise action boundary and regenerate trustworthy E015 videos and phase-grid metrics.

**Architecture:** A pure evaluator helper owns the action-boundary rule. The rollout calls it once after optional noise and before the compiled environment step, preserving raw and effective evidence separately.

**Tech Stack:** Python, JAX, NumPy, pytest, MuJoCo/MJX.

## Global Constraints

- Do not alter training behavior or retrain E015.
- Clip exactly to `[-1, 1]` only when `squash_actor_actions` is true.
- Preserve unbounded action-parity environments exactly.
- Corrected evidence must use the pinned E015 checkpoint, reference, solver, and phase grid.

---

### Task 1: Action-Boundary Regression

**Files:**
- Modify: `tools/evaluate_g1_tracking.py`
- Test: `tests/test_g1_tracking_evaluator.py`

**Interfaces:**
- Produces: `prepare_evaluation_action(action, *, squash: bool) -> jax.Array`
- Consumes: raw post-noise actor actions and `env.squash_actor_actions`

- [ ] **Step 1: Write the failing tests**

Add tests asserting that `[-2.0, 0.25, 2.0]` becomes
`[-1.0, 0.25, 1.0]` when squash is true and remains unchanged when false.

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest -q tests/test_g1_tracking_evaluator.py -k prepare_evaluation_action`

Expected: FAIL because `prepare_evaluation_action` is absent.

- [ ] **Step 3: Implement the minimal helper and rollout wiring**

Define the pure helper with `jnp.clip` for the bounded case. Invoke it after
optional training-noise injection and before `compiled_step(state, action)`.
When publishing training-noise evidence, add the effective action array while
retaining the raw mean, epsilon, and noisy action arrays.

- [ ] **Step 4: Run focused verification**

Run:
`/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest -q tests/test_g1_tracking_evaluator.py tests/test_g1_action_noise_pair.py`

Expected: all tests pass.

- [ ] **Step 5: Run static checks and commit**

Run Ruff on the two changed Python files, `py_compile` the evaluator, and
`git diff --check`. Commit the spec, plan, test, and implementation together as
`fix: match G1 evaluation action boundary`.

### Task 2: Correct E015 Evidence

**Files:**
- Create only immutable run artifacts under the E015 run directory.
- Update canonical experiment/state records only after validation.

**Interfaces:**
- Consumes: E015 selected checkpoint step 1,966,080, fixed reference, solver
  `g1-4x5`, phases `0/100/200/300/400`, seed zero.
- Produces: corrected clean/noisy phase-zero videos, summaries, and corrected
  phase-grid survival vector.

- [ ] **Step 1: Render corrected phase-zero rollouts**

Run the existing evaluator twice from the repaired clean commit: once nominally
and once with `--training-distribution-rollout` using the E015 hparams.

- [ ] **Step 2: Evaluate the corrected five-phase grid**

Use the existing phase-grid evaluator with the selected checkpoint/reference
hashes and solver profile. Store results in a new correction directory; never
overwrite the original artifacts.

- [ ] **Step 3: Validate and curate**

Verify action bounds, provenance, finite telemetry, terminal consistency,
videos, montage, and phase-grid summaries. If the corrected result changes the
registered verdict, supersede the old claim in the E015 result and current
state, rebuild the vault, validate the registry, and commit only canonical
record changes.

