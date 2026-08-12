# G1 Assistance-Curriculum Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repeat E012 from the exact E008 checkpoint under an independently rekeyed training-randomness stream while preserving every non-random resume leaf.

**Architecture:** Add one pure resume-rekey module at the SHAC boundary, wire it through the default-off `train` API after all migrations and before device commitment, and expose one immutable confirmation runner that is an exact E012 delta. Persist a compact audit proving that only trainer and environment RNG keys changed.

**Tech Stack:** Python 3.11, JAX, Flax TrainState PyTrees, pytest, Ruff, existing Open-DiffLoco SHAC runners.

## Global Constraints

- `resume_random_seed` defaults to `None`; existing exact resume remains unchanged.
- Confirmation seed is exactly 1; ordinary runner seed remains exactly 0.
- Only `TrainState.key` and `env_state.info["rng"]` may change at the resume boundary.
- Resume exact E008 step 1,327,104 and preserve the complete E012 schedule and endpoint.
- Do not alter E012 artifacts, checkpoint selection, actor dimension, objective, solver, reward, or evaluation protocol.

---

### Task 1: Pure resume-randomness boundary

**Files:**
- Create: `src/algorithms/shac/resume_randomness.py`
- Create: `tests/test_shac_resume_randomness.py`

**Interfaces:**
- Consumes: `TrainState` with `env_state.info["rng"]` shaped `(num_envs, 2)`.
- Produces: `rekey_resumed_train_state(state, *, seed: int) -> TrainState` and `resume_randomness_audit(original, candidate, *, seed: int) -> dict[str, object]`.

- [ ] **Step 1: Write failing deterministic, preservation, and validation tests**

Construct a small synthetic `TrainState` whose environment info contains four
PRNG keys. Assert seed 1 changes the top-level and environment keys, repeated
seed 1 calls are exact, seed 2 differs, and all non-RNG leaves remain exact.
Parametrize invalid seeds `True`, `-1`, and `1.5`; add missing and malformed RNG
leaf cases.

- [ ] **Step 2: Run the RED slice**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_resume_randomness.py
```

Expected: collection fails because `src.algorithms.shac.resume_randomness` does
not exist.

- [ ] **Step 3: Implement the minimal pure helper and audit**

Use fixed namespace constants, `jax.random.PRNGKey`, `jax.random.fold_in`, and
`jax.random.split`. Rebuild only `state.key` and the copied info mapping's
`"rng"` leaf. The audit hashes NumPy key bytes and verifies exact equality of a
copy with the two authorized leaves restored to their original values.

- [ ] **Step 4: Run the GREEN slice and static checks**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_resume_randomness.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check src/algorithms/shac/resume_randomness.py tests/test_shac_resume_randomness.py
```

Expected: all focused tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/algorithms/shac/resume_randomness.py tests/test_shac_resume_randomness.py
git commit -m "feat: add exact resume randomness rekey"
```

### Task 2: Trainer integration and audit persistence

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tests/test_shac_exact_resume.py`
- Test: `tests/test_shac_resume_randomness.py`

**Interfaces:**
- Consumes: `resume_random_seed: int | None` on `train`.
- Produces: persisted `resume_random_seed` hparam and `resume_randomness_audit.json` for opted-in resumes.

- [ ] **Step 1: Write failing API and integration tests**

Assert the train signature defaults `resume_random_seed` to `None`, source
wiring invokes the pure helper only for a resumed state, and a small persistence
test writes an atomic JSON audit with `non_rng_state_exact: true`.

- [ ] **Step 2: Run the RED integration slice**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_resume_randomness.py tests/test_shac_exact_resume.py
```

Expected: new signature/wiring assertions fail.

- [ ] **Step 3: Integrate after resume migrations and before device commit**

Add `resume_random_seed: int | None = None` to `train`. Reject non-`None` use
without `resume_from`. Immediately before `select_initial_training_state`, keep
the pre-rekey state, call `rekey_resumed_train_state`, build the audit, and
atomically persist it under the run directory. Include the setting in
`hparams.json` and describe the rekey explicitly in the resume log.

- [ ] **Step 4: Run focused and neighboring tests**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_shac_resume_randomness.py \
  tests/test_shac_exact_resume.py \
  tests/test_shac_future_reference_migration.py \
  tests/test_shac_torso_wrench_curriculum.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
  src/algorithms/shac/algorithm.py \
  src/algorithms/shac/resume_randomness.py \
  tests/test_shac_resume_randomness.py
```

Expected: all selected tests and Ruff pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_exact_resume.py tests/test_shac_resume_randomness.py
git commit -m "feat: wire independent randomness into exact resume"
```

### Task 3: Immutable confirmation runner

**Files:**
- Create: `tools/run_g1_frozen_residual_assistance_confirmation.py`
- Create: `tests/test_g1_frozen_residual_assistance_confirmation_runner.py`

**Interfaces:**
- Consumes: exact E012 kwargs builder.
- Produces: `build_frozen_residual_assistance_confirmation_kwargs(...) -> dict` with the sole delta `resume_random_seed=1`.

- [ ] **Step 1: Write the failing exact-delta runner test**

Build E012 and confirmation kwargs from identical dummy paths and assert:

```python
delta = {
    key: value
    for key, value in confirmation.items()
    if baseline.get(key) != value
}
assert delta == {"resume_random_seed": 1}
```

Also assert the parser retains ordinary `--seed 0` and requires the same resume,
reference, solver, and output arguments.

- [ ] **Step 2: Run RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_frozen_residual_assistance_confirmation_runner.py
```

Expected: module import fails.

- [ ] **Step 3: Implement the thin runner**

Delegate all settings to `build_frozen_residual_assistance_kwargs`, add only
`resume_random_seed=1`, and reuse the established solver/output-root execution
pattern. Do not add new flags or schedule constants.

- [ ] **Step 4: Verify runner and parent contracts**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_frozen_residual_assistance_confirmation_runner.py \
  tests/test_g1_frozen_residual_assistance_runner.py \
  tests/test_shac_resume_randomness.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
  tools/run_g1_frozen_residual_assistance_confirmation.py \
  tests/test_g1_frozen_residual_assistance_confirmation_runner.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
  tools/run_g1_frozen_residual_assistance_confirmation.py
git diff --check
```

Expected: all selected tests and static checks pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add tools/run_g1_frozen_residual_assistance_confirmation.py tests/test_g1_frozen_residual_assistance_confirmation_runner.py
git commit -m "feat: add E012 randomness confirmation runner"
```

### Task 4: Review, register, and execute confirmation

**Files:**
- Create in canonical lab: `research/experiments/E-20260811-013.yaml`
- Modify after result only: `research/state/current.yaml`
- Generated after result: `docs/vault/experiments/E-20260811-013.md`, `docs/vault/02 Current State.md`, `docs/vault/hypotheses/H-SCI-001.md`

**Interfaces:**
- Consumes: reviewed exact code commit and E008 checkpoint/reference/model/controller hashes.
- Produces: one guarded seed-zero invocation whose internal resume randomness seed is exactly one.

- [ ] **Step 1: Run final code verification and request independent review**

Run the focused suite from Tasks 1–3, inspect the exact diff, and require zero
Critical/Important findings on the rekey authority, preservation audit, and
runner delta before registration.

- [ ] **Step 2: Write and validate E013**

Copy E012's provenance and gates, change the code commit, runner command,
required rekey audit, question, outcome map, and descriptive seed comparison.
Pin one GPU, seed list `[0]`, internal resume seed 1, and a 120-minute budget.

Run:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260811-013 --dry-run
```

Expected: registry valid and dry-run command exactly matches the confirmation
runner with E008 resume.

- [ ] **Step 3: Approve and launch one guarded run**

Set `approved_by: codex` with UTC time, commit the registration, then run:

```bash
uv run python -m tools.runexp E-20260811-013
```

Continue only after the first update has finite gradients, valid CAGrad,
preview, and wrench telemetry, five occupied bins, and a plausible active
fraction.

- [ ] **Step 4: Evaluate all checkpoints and curate**

Run the unchanged five-phase zero-wrench evaluator for all eight checkpoints,
write the minimum/median/mean/earliest selection artifact, verify every required
artifact and hash, and select exactly one preregistered outcome.

- [ ] **Step 5: Rebuild, test, commit, and publish research evidence**

Run:

```bash
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py build-vault
uv run pytest -q tests/test_registry.py
git diff --check
```

Commit only E013 registry/state/generated projections, preserve user-owned
Obsidian settings and canvas files, and push the code branch plus canonical lab
main.
