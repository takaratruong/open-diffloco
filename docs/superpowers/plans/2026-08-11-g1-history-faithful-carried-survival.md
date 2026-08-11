# G1 History-Faithful Carried Survival Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the selected E008 residual policy from exact policy-carried pre-failure states whose full ten-frame actor context is restored, then evaluate whether this raises replay-free five-phase survival.

**Architecture:** Extend the existing opt-in carried-reset boundary so context-rich banks restore `last_act` and `actor_obs_history` together with `qpos/qvel/phase`, while legacy banks remain unchanged. A standalone collector materializes and validates an immutable 120-row bank from exact E008 rollouts. A strict resume resolver and immutable continuation runner make the reset distribution the sole training change.

**Tech Stack:** Python 3.10, NumPy, JAX, MJX, Flax, Optax, pytest, Ruff.

## Global Constraints

- Parent checkpoint is exact E008 step 1,327,104 with SHA-256 `fbea5e272d1431c08753a3600014623cd5577e34e01aeeba18b16af46d369377`.
- Reference SHA-256 is `bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db`.
- Source starts are exactly `0/100/200/300/400`, with expected E008 survival `70/63/95/70/44`.
- Retain states 6 through 29 transitions before each terminal transition: exactly 24 rows per source and 120 rows total.
- Carried reset probability is exactly 0.5; bank start is 0.
- Training is exactly 64 effective-512 H12 updates: 393,216 transitions from step 1,327,104 to 1,720,320.
- Checkpoint interval is 49,152 transitions, producing eight treatment checkpoints.
- Preserve Adam, five-bin CAGrad, frozen parent actor, trained nonlinear delta-preview residual, actor normalizer, critic, fixed `g1-4x5` solver, reward, action-noise schedule, seed 0, and nominal physics.
- Do not bind any server to `0.0.0.0`.

---

### Task 1: Restore Complete Actor Context From Carried Banks

**Files:**
- Modify: `src/envs/g1_tracking/environment.py:292-302,398-458,746-902,967-1017`
- Modify: `tests/test_g1_tracking_environment.py:493-560`

**Interfaces:**
- Consumes: NPZ arrays `qpos: (N,36)`, `qvel: (N,35)`, `phase: (N,)`, and optional paired arrays `last_act: (N,29)`, `actor_obs_history: (N,10,328)`.
- Produces: `G1TrackingEnv.carried_reset_restores_actor_context: bool`; `_initial_state_from_data(..., last_act=None, actor_obs_history=None) -> EnvState`.

- [ ] **Step 1: Write failing context-restoration tests**

Add a test that constructs a two-row bank from valid reference states, supplies distinct finite `last_act` and `(2, 10, actor_frame_obs_dim)` histories, starts at row one with probability one, and asserts exact restoration:

```python
np.testing.assert_allclose(state.info["last_act"], last_act[1], rtol=0, atol=0)
np.testing.assert_allclose(
    state.info["actor_obs_history"], history[1], rtol=0, atol=0
)
np.testing.assert_allclose(state.obs, history[1].reshape(-1), rtol=0, atol=0)
np.testing.assert_allclose(
    state.info["bootstrap_obs"], history[1].reshape(-1), rtol=0, atol=0
)
self.assertTrue(env.carried_reset_restores_actor_context)
```

Also test that only one of the two context arrays raises `ValueError`, wrong shapes and non-finite context raise `ValueError`, probability-zero reference resets retain zero `last_act` and repeated history, and the existing three-array legacy bank test remains green.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
JAX_ENABLE_X64=1 micromamba run -n diffsim python -m pytest \
  tests/test_g1_tracking_environment.py \
  -k 'carried_reset' -q
```

Expected: the new context tests fail because the environment neither loads nor restores the arrays; legacy tests pass.

- [ ] **Step 3: Implement paired context loading and restoration**

At bank load, require both context keys or neither, validate exact shapes against `action_dim`, `actor_history_len`, and `actor_frame_obs_dim`, validate finiteness, apply the same bank slice, and set:

```python
self.carried_reset_restores_actor_context = has_context
self.carried_reset_last_act = (
    jp.asarray(carried_last_act[bank_slice]) if has_context else None
)
self.carried_reset_actor_obs_history = (
    jp.asarray(carried_history[bank_slice]) if has_context else None
)
```

Allow `_initial_state_from_data` to receive optional context. Set `info["last_act"]` before computing the actor frame and critic observation. When an explicit history is supplied, use it instead of repeating the current frame and set both `obs` and `bootstrap_obs` from that history.

In the carried-reset branch, select one bank index for all arrays. For a context bank, use `jax.numpy.where(use_carried, bank_last_act, zeros)` and first build the physical state with the matched previous action. Then choose between the stored carried history and the freshly constructed reference history and update `obs`, `actor_obs_history`, and `bootstrap_obs` together. A legacy bank follows the old zero-action/repeated-history path.

- [ ] **Step 4: Run focused and neighboring environment tests**

Run:

```bash
JAX_ENABLE_X64=1 micromamba run -n diffsim python -m pytest \
  tests/test_g1_tracking_environment.py -q
micromamba run -n diffsim ruff check \
  src/envs/g1_tracking/environment.py tests/test_g1_tracking_environment.py
```

Expected: all environment tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit the environment boundary**

```bash
git add src/envs/g1_tracking/environment.py tests/test_g1_tracking_environment.py
git commit -m "feat: restore carried actor context"
```

---

### Task 2: Build And Validate The Immutable E008 Bank

**Files:**
- Create: `tools/build_g1_history_carried_reset_bank.py`
- Create: `tests/test_g1_history_carried_reset_bank.py`

**Interfaces:**
- Produces: `select_preterminal_indices(step_count: int, min_remaining: int = 6, max_remaining: int = 29) -> numpy.ndarray`.
- Produces: `validate_history_bank(arrays: Mapping[str, numpy.ndarray], *, expected_source_phases: tuple[int, ...], expected_survival: tuple[int, ...], history_len: int, frame_dim: int) -> dict[str, object]`.
- Produces CLI artifacts: atomically written `carried_reset_bank.npz` and `carried_reset_bank.json` with source hashes, exact survival, row provenance, schema version, and bank SHA-256.

- [ ] **Step 1: Write failing pure selector and schema tests**

Use synthetic rollouts with survival `(70,63,95,70,44)`. Assert each source yields exactly 24 indices, `transitions_to_terminal` is exactly `29,28,...,6`, total rows are 120, phases advance by one, and all required arrays remain row-aligned. Reject wrong source survival, terminal source rows, hard-limit violations, non-finite arrays, non-normalized quaternions, mismatched history dimensions, and a last history frame inconsistent with the stored fresh actor frame.

- [ ] **Step 2: Run the new module test and confirm RED**

```bash
micromamba run -n diffsim python -m pytest \
  tests/test_g1_history_carried_reset_bank.py -q
```

Expected: import failure because the collector module does not exist.

- [ ] **Step 3: Implement pure selection, validation, and atomic writers**

The selector is:

```python
remaining = np.arange(step_count, 0, -1, dtype=np.int32)
return np.flatnonzero((remaining >= min_remaining) & (remaining <= max_remaining))
```

The validator requires these aligned fields:

```python
required = {
    "qpos": (rows, 36),
    "qvel": (rows, 35),
    "phase": (rows,),
    "last_act": (rows, 29),
    "actor_obs_history": (rows, history_len, frame_dim),
    "fresh_actor_frame": (rows, frame_dim),
    "action": (rows, 29),
    "source_start_phase": (rows,),
    "source_step": (rows,),
    "transitions_to_terminal": (rows,),
    "termination_errors": (rows, 4),
    "termination_thresholds": (4,),
}
```

Require exact equality between `actor_obs_history[:, -1]` and
`fresh_actor_frame`, strictly positive hard-limit clearance, integer phase and
provenance arrays, normalized root quaternions, and exact per-source offset and
survival contracts. Write NPZ to an opened temporary binary file before
`os.replace`; compute SHA-256 by streaming blocks for Python 3.10 compatibility.

- [ ] **Step 4: Implement exact E008 rollout collection**

Reuse `configure_jax`, `make_evaluation_env`, `evaluate_actor_action`,
`FrozenPreviewResidualParams`, and `PreviewResidualAdapter`. Build the exact
delta-preview, history-10, residual-256 actor used by the phase-grid evaluator.
Before each environment step, materialize `qpos`, `qvel`, `phase`, `last_act`,
history, fresh actor frame, action, and hard-limit errors. After the rollout,
require the exact registered terminal count, then select offsets 6 through 29.
Collect all five sources within one `solver_context(g1-4x5)` process so JAX
compilation is reused.

- [ ] **Step 5: Run tests and static checks**

```bash
micromamba run -n diffsim python -m pytest \
  tests/test_g1_history_carried_reset_bank.py \
  tests/test_evaluate_g1_flax_phase_grid.py -q
micromamba run -n diffsim ruff check \
  tools/build_g1_history_carried_reset_bank.py \
  tests/test_g1_history_carried_reset_bank.py
micromamba run -n diffsim python -m py_compile \
  tools/build_g1_history_carried_reset_bank.py
```

Expected: all tests and static checks pass.

- [ ] **Step 6: Commit the collector**

```bash
git add tools/build_g1_history_carried_reset_bank.py \
  tests/test_g1_history_carried_reset_bank.py
git commit -m "feat: collect history-faithful carried states"
```

---

### Task 3: Add A Strict Resume Treatment And Immutable Continuation Runner

**Files:**
- Modify: `src/algorithms/shac/algorithm.py:735-770,1000-1085,1210-1265,1360-1430,2970-3010`
- Modify: `tests/test_shac_exact_resume.py:300-340`
- Create: `tools/run_g1_frozen_residual_carried_continuation.py`
- Create: `tests/test_g1_frozen_residual_carried_continuation_runner.py`

**Interfaces:**
- Produces: `resolve_carried_reset_resume_settings(resumed_hparams, *, requested_bank_path, requested_probability, requested_start, allow_change) -> tuple[str | None, float, int]`.
- Produces: `build_frozen_residual_carried_kwargs(profile_name, reference_path, seed, resume_from, carried_bank) -> dict`.

- [ ] **Step 1: Write failing resume-boundary tests**

Assert that an unchanged resumed bank is restored, a fresh run accepts requested
settings, a changed bank/probability/start is rejected unless `allow_change` is
true, and a boolean or non-boolean allow flag is handled strictly. The dedicated
runner must set exactly:

```python
{
    "total_steps": 1_720_320,
    "checkpoint_interval": 49_152,
    "actor_residual_preview_adapter": True,
    "actor_residual_preview_hidden": 256,
    "actor_residual_preview_optimizer": "adam",
    "carried_reset_bank_path": str(bank.resolve()),
    "carried_reset_probability": 0.5,
    "carried_reset_bank_start": 0,
    "allow_resume_carried_reset_change": True,
}
```

Also assert the parser requires `--carried-reset-bank` and
`--carried-reset-bank-sha256`, and rejects a hash mismatch before calling
`train`.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
micromamba run -n diffsim python -m pytest \
  tests/test_shac_exact_resume.py \
  tests/test_g1_frozen_residual_carried_continuation_runner.py -q
```

Expected: new helper and runner import failures.

- [ ] **Step 3: Implement the strict resume resolver**

Add `allow_resume_carried_reset_change: bool = False` to `train`. Treat an all-
default request `(None, 0.0, 0)` as “restore checkpoint settings.” Otherwise,
require an exact checkpoint match unless the allow flag is true. When allowed,
return the requested settings rather than overwriting them from resumed hparams.
Call this resolver before environment construction and record the allow flag in
`hparams.json`. Preserve existing behavior for fresh and legacy runs.

- [ ] **Step 4: Implement the immutable E010 runner**

Build from `build_frozen_residual_preview_kwargs`, override only the fixed values
above, stream-check the requested NPZ SHA before configuring JAX, and then call
`train` inside `solver_context(g1-4x5)`. The runner accepts no CLI knobs for
probability, offsets, horizon, optimizer, batch, CAGrad, or budget.

- [ ] **Step 5: Run focused and neighboring tests**

```bash
micromamba run -n diffsim python -m pytest \
  tests/test_shac_exact_resume.py \
  tests/test_g1_frozen_residual_preview_runner.py \
  tests/test_g1_frozen_residual_carried_continuation_runner.py \
  tests/test_shac_microbatch.py \
  tests/test_shac_cagrad_integration.py -q
micromamba run -n diffsim ruff check \
  src/algorithms/shac/algorithm.py \
  tools/run_g1_frozen_residual_carried_continuation.py \
  tests/test_shac_exact_resume.py \
  tests/test_g1_frozen_residual_carried_continuation_runner.py
micromamba run -n diffsim python -m py_compile \
  src/algorithms/shac/algorithm.py \
  tools/run_g1_frozen_residual_carried_continuation.py
```

Expected: all tests and static checks pass.

- [ ] **Step 6: Commit the resume boundary and runner**

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_exact_resume.py \
  tools/run_g1_frozen_residual_carried_continuation.py \
  tests/test_g1_frozen_residual_carried_continuation_runner.py
git commit -m "feat: add carried-survival continuation"
```

---

### Task 4: Materialize, Smoke, Register, Run, And Evaluate E010

**Files:**
- Create at runtime: canonical E010 carried bank NPZ and JSON under `$DIFFSIM_RUN_DIR`
- Create after green code review: `research/experiments/E-20260811-010.yaml` in `/home/ubuntu/projects/diffsim2real-lab`
- Modify after evaluation: `research/state/current.yaml`
- Create after evaluation: a decision or finding only if the evidence settles a reusable claim

**Interfaces:**
- Consumes: exact E008 selected checkpoint and the committed collector/runner.
- Produces: one guarded E010 run, eight dense replay-free phase summaries, frozen-state audit, context-reset audit, and deterministic selection artifact.

- [ ] **Step 1: Run the complete focused regression suite**

```bash
JAX_ENABLE_X64=1 micromamba run -n diffsim python -m pytest \
  tests/test_g1_tracking_environment.py \
  tests/test_g1_history_carried_reset_bank.py \
  tests/test_shac_exact_resume.py \
  tests/test_g1_frozen_residual_carried_continuation_runner.py \
  tests/test_g1_frozen_residual_preview_runner.py \
  tests/test_shac_cagrad.py \
  tests/test_shac_cagrad_integration.py \
  tests/test_shac_microbatch.py -q
```

Expected: all selected tests pass with x64 enabled.

- [ ] **Step 2: Materialize and validate the canonical bank on one L40S**

Run the committed collector with the exact checkpoint/reference paths and hashes,
source starts, survival vector, residual adapter, delta preview, and `g1-4x5`.
Require JSON `valid: true`, 120 rows, 24 rows per source, minimum hard-limit
clearance above zero, exact history-frame agreement, and bank SHA-256 matching
the NPZ bytes. Preserve both artifacts immutably.

- [ ] **Step 3: Run a one-update JIT smoke from E008**

Invoke `train` with the exact runner kwargs except `total_steps=1_333_248` and
`checkpoint_interval=6_144`. Require one effective-512 H12 update, finite actor
and critic gradients, nonzero residual update, unchanged frozen parent and actor
normalizer, exact bank SHA in hparams, and a probability-one reset audit that
reproduces stored `qpos/qvel/phase/last_act/history`.

- [ ] **Step 4: Review the scientific diff and commit any bounded repair**

Inspect `git diff`, bank metadata, smoke hparams, and frozen/context audits. Any
failure invalidates the launch. Repair only the failed implementation boundary,
rerun its focused test and the one-update smoke, then commit one bounded repair.

- [ ] **Step 5: Register and launch E010**

Pin the clean code commit, checkpoint/reference/bank paths and hashes, GPU UUID,
64-update budget, checkpoint grid, exact control vector, advancement gates, and
all invalidity conditions in the canonical experiment record. Run
`uv run python tools/researchctl.py validate`, commit approval, then execute the
single guarded training attempt.

- [ ] **Step 6: Evaluate all eight checkpoints replay-free**

Run the unchanged five-phase Flax evaluator on every checkpoint. Build one
selection artifact ordered by minimum survival, median, mean, then earliest
step. Independently audit numeric finiteness, frozen parent and normalizer
identity, authorized residual changes, bank identity, context restoration, exact
transition/update counts, and absence of timeout.

- [ ] **Step 7: Curate the result and choose the next causal action**

Advance only if the selected vector is componentwise at least
`70/63/95/70/44` and lexicographically exceeds `44/70/68.4`. If valid but below
the gate, retain E008 and design the deferred terminal-value intervention. If
invalid, repair only the invalid boundary. Update canonical state, rebuild the
vault, validate the registry, run focused registry tests, and commit the curated
result without touching user-owned Obsidian settings or canvas files.
