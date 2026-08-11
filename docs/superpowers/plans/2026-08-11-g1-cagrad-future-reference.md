# G1 CAGrad Future-Reference Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off +4/+8/+12 control-step reference preview to the G1 actor, migrate E008 without changing its pre-update actions, and run one pinned H12 CAGrad continuation.

**Architecture:** The G1 environment expands each 154-value actor frame with three 58-value future joint commands. A focused checkpoint module expands E008's ten-frame actor kernel, Adam moments, normalizer, and saved observation histories while proving pre-update action equivalence. SHAC opts into this one append-only migration at resume; an immutable runner and explicit evaluator flags keep the scientific treatment pinned.

**Tech Stack:** Python 3.11, JAX, Flax, Optax, MuJoCo MJX, unittest/pytest, Ruff, YAML research registry.

## Global Constraints

- The registered lookahead tuple is exactly `(4, 8, 12)` control steps.
- E008 history remains exactly ten frames; actor input grows from 1,540 to 3,280 values.
- Resume exact E008 SHA-256 `da661e4bec6fbbd578face31ea2129e00c01d985cc6308e20473db4d9055fe3c` at step 1,179,648.
- Preserve H12, physical environments 256, accumulation factor two, effective batch 512, five-bin CAGrad, fixed 4x5 solver, and seed zero.
- Run exactly 64 further actor updates and 393,216 transitions, ending at step 1,572,864 with checkpoints at 1,376,256 and 1,572,864.
- Critic observation remains 286 values and all new behavior defaults off.
- Unsupported checkpoint shape changes fail closed; never fall back to fresh initialization.
- Advancement requires minimum survival at least 19, median at least 48, and phase vector at least `23/50/72/16/40`.

---

## File Structure

**Open-DiffLoco worktree** `/home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805`:

- Modify `src/envs/g1_tracking/environment.py`: validate lookahead offsets, construct preview commands, expand frame dimensions and noise masks.
- Modify `src/envs/g1_tracking/training_distribution.py`: preserve actor corruption semantics for expanded frames.
- Modify `tests/test_g1_tracking_environment.py`: environment indexing, clamping, ordering, history, reset, and default-off tests.
- Modify `tests/test_g1_tracking_training_distribution.py`: noise-suffix and expanded-reset tests.
- Create `src/algorithms/shac/future_reference_migration.py`: pure append-only E008 TrainState migration and equivalence report.
- Create `tests/test_shac_future_reference_migration.py`: synthetic exactness, optimizer, history, metadata, and failure tests.
- Modify `src/algorithms/shac/algorithm.py`: resume resolution, environment wiring, migration, artifact, and hparams.
- Modify `tests/test_shac_exact_resume.py`: treatment-resolution and exact-resume tests.
- Create `tools/run_g1_cagrad_future_reference_continuation.py`: immutable E008 successor.
- Create `tests/test_g1_cagrad_future_reference_runner.py`: exact contract and CLI tests.
- Modify `tools/evaluate_g1_tracking.py`: explicit lookahead evaluator option.
- Modify `tools/evaluate_g1_phase_grid.py`: forward the pinned lookahead tuple.
- Modify `tests/test_g1_tracking_evaluator.py`: evaluator environment test.
- Modify `tests/test_g1_phase_grid_evaluator.py`: command-contract test.

**Canonical research repo** `/home/ubuntu/projects/diffsim2real-lab`:

- Create `research/experiments/E-20260811-000.yaml`: preregister the pinned continuation and evaluation gates.
- Modify `research/state/current.yaml`: point the active decision to E010 only after approval/launch, then curate validated results.
- Regenerate `docs/vault/generated/` through `tools/researchctl.py build-vault`; never edit generated notes directly.

---

### Task 1: Expanded G1 Future-Reference Observation

**Files:**
- Modify: `src/envs/g1_tracking/environment.py:120-180,470-515,625-642,1382-1402`
- Modify: `src/envs/g1_tracking/training_distribution.py:177-202`
- Modify: `tests/test_g1_tracking_environment.py:561-580`
- Modify: `tests/test_g1_tracking_training_distribution.py:95-145,205-250`

**Interfaces:**
- Produces: `G1TrackingEnv(..., actor_reference_lookahead_steps: tuple[int, ...] = ())`
- Produces: `G1TrackingEnv._future_reference_command(phase: jax.Array) -> jax.Array`
- Produces: `corrupt_actor_observation(key, observation, *, legacy_frame_dim: int = 154) -> jax.Array`
- Preserves: default frame dimension 154, critic dimension 286, and existing field ordering.

- [ ] **Step 1: Write failing environment contract tests**

Add tests that instantiate one default environment and one treatment environment with history ten. Verify the treatment frame's first 154 values exactly equal the default frame, then compare its suffix to explicit reference gathers:

```python
treatment = G1TrackingRMR50HzValidatedEnv(
    xml_path=str(MODEL),
    reference_path=str(REFERENCE),
    controller_path=str(CONTROLLER),
    actor_history_len=10,
    actor_reference_lookahead_steps=(4, 8, 12),
)
state = treatment.reset_at_phase(
    jax.random.PRNGKey(71), jnp.array(0.0), jnp.array(37)
)
frames = np.asarray(state.obs).reshape(10, 328)
expected = []
for offset in (4, 8, 12):
    index = min(37 + offset * treatment.reference_stride,
                treatment.reference_length - 1)
    expected.extend(np.asarray(treatment.qpos_reference[index, 7:])[
        np.asarray(treatment.model_to_actor_permutation)
    ])
    expected.extend(np.asarray(treatment.qvel_reference[index, 6:])[
        np.asarray(treatment.model_to_actor_permutation)
    ])
np.testing.assert_allclose(frames[-1, 154:], expected, atol=0.0, rtol=0.0)
self.assertEqual(treatment.actor_frame_obs_dim, 328)
self.assertEqual(treatment.actor_obs_dim, 3280)
self.assertEqual(treatment.critic_obs_dim, 286)
```

Also test final-frame clamping, strict positive/increasing/unique integer validation, step/bootstrap/reset shapes, and default-off byte equality.

- [ ] **Step 2: Run the focused tests to establish RED**

Run:

```bash
conda run -n diffsim python -m pytest \
  tests/test_g1_tracking_environment.py \
  tests/test_g1_tracking_training_distribution.py -q
```

Expected: new tests fail because the constructor option and expanded noise contract do not exist; existing tests remain green.

- [ ] **Step 3: Implement the minimal environment treatment**

In `G1TrackingEnv.__init__`, canonicalize the tuple and derive dimensions:

```python
offsets = tuple(actor_reference_lookahead_steps)
if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in offsets):
    raise ValueError("actor reference lookahead steps must be positive integers")
if tuple(sorted(set(offsets))) != offsets:
    raise ValueError("actor reference lookahead steps must be strictly increasing")
self.actor_reference_lookahead_steps = offsets
self.actor_future_reference_dim = 58 * len(offsets)
self.actor_frame_obs_dim = 154 + self.actor_future_reference_dim
self.actor_obs_dim = self.actor_frame_obs_dim * actor_history_len
self.actor_noise_mask = jp.concatenate(
    (legacy_actor_noise_mask, jp.zeros(self.actor_future_reference_dim))
)
```

Implement one helper and append it at the end of `_get_actor_obs`:

```python
def _future_reference_command(self, phase):
    if not self.actor_reference_lookahead_steps:
        return jp.zeros((0,), dtype=self.qpos_reference.dtype)
    offsets = jp.asarray(self.actor_reference_lookahead_steps, dtype=jp.int32)
    indices = jp.minimum(
        phase + offsets * self.reference_stride,
        self.reference_length - 1,
    )
    order = self.model_to_actor_permutation
    commands = jp.concatenate(
        (self.qpos_reference[indices, 7:][:, order],
         self.qvel_reference[indices, 6:][:, order]),
        axis=-1,
    )
    return commands.reshape(-1)
```

Generalize `corrupt_actor_observation` to require `dimension >= 154` and `(dimension - 154) % 58 == 0`, while applying the existing slices only. Do not alter noise values or PRNG splitting.

- [ ] **Step 4: Run focused GREEN and legacy-default tests**

Run the Step 2 command and:

```bash
conda run -n diffsim python -m pytest \
  tests/test_canonical_g1_shac_smoke.py \
  tests/test_canonical_g1_shac_runner.py -q
```

Expected: all selected tests pass; treatment observations are 3280 and legacy observations remain 1540.

- [ ] **Step 5: Commit the environment boundary**

```bash
git add src/envs/g1_tracking/environment.py \
  src/envs/g1_tracking/training_distribution.py \
  tests/test_g1_tracking_environment.py \
  tests/test_g1_tracking_training_distribution.py
git commit -m "feat: add G1 future reference observations"
```

### Task 2: Behavior-Preserving E008 Checkpoint Migration

**Files:**
- Create: `src/algorithms/shac/future_reference_migration.py`
- Create: `tests/test_shac_future_reference_migration.py`

**Interfaces:**
- Consumes: a legacy `TrainState`, a freshly initialized expanded `TrainState`, and the treatment environment.
- Produces: `migrate_future_reference_train_state(resumed_state, initialized_state, env, *, expected_history_len: int = 10) -> TrainState`
- Produces: `future_reference_migration_report(original_state, migrated_state, actor, legacy_env, treatment_env) -> dict[str, object]`
- Produces: `validate_future_reference_migration_report(report) -> None`

- [ ] **Step 1: Write failing synthetic migration tests**

Build legacy/treatment actors with input sizes 1540/3280 and Optax Adam states. Seed nonzero legacy kernels and moments. Assert the block mapping explicitly:

```python
old = original.actor_params["params"]["Dense_0"]["kernel"].reshape(10, 154, 512)
new = migrated.actor_params["params"]["Dense_0"]["kernel"].reshape(10, 328, 512)
np.testing.assert_array_equal(new[:, :154], old)
np.testing.assert_array_equal(new[:, 154:], 0.0)
np.testing.assert_array_equal(
    migrated.actor_opt[1][0].mu["params"]["Dense_0"]["kernel"].reshape(10, 328, 512)[:, 154:],
    0.0,
)
```

Assert legacy normalizer entries and all critic/non-actor leaves are exact, new mean/variance are finite, env observations/history/bootstrap have expanded shapes, and actor outputs agree within `1e-7`. Add failures for history other than ten, nonempty saved lookahead metadata, unexpected kernel shapes, and optimizer-template mismatch.

- [ ] **Step 2: Run migration tests to establish RED**

```bash
conda run -n diffsim python -m pytest \
  tests/test_shac_future_reference_migration.py -q
```

Expected: collection fails because `future_reference_migration` does not exist.

- [ ] **Step 3: Implement blockwise parameter and optimizer expansion**

Use a pure helper for all input-row matrices:

```python
def expand_history_input_rows(array, *, history_len, old_frame_dim, new_frame_dim):
    value = jp.asarray(array)
    if value.shape[0] != history_len * old_frame_dim:
        raise ValueError("legacy input-row shape does not match declared history")
    blocks = value.reshape((history_len, old_frame_dim) + value.shape[1:])
    expanded = jp.zeros(
        (history_len, new_frame_dim) + value.shape[1:], dtype=value.dtype
    )
    return expanded.at[:, :old_frame_dim].set(blocks).reshape(
        (history_len * new_frame_dim,) + value.shape[1:]
    )
```

Apply it only to `Dense_0/kernel` in actor params and the matching `mu`/`nu` leaves in `actor_opt[1][0]`. Copy every other saved optimizer leaf after verifying it matches the initialized expanded template.

- [ ] **Step 4: Implement normalizer and environment-state expansion**

Compute suffix statistics by vmapping `env._future_reference_command` over every reference phase, then append its mean and variance to the saved 154-value normalizer. Expand each saved history block by preserving the 154 legacy values and filling the suffix using phases inferred as:

```python
age = jp.arange(history_len - 1, -1, -1, dtype=jp.int32)
effective_age = jp.minimum(age, current_episode_step)
historical_phase = jp.maximum(
    current_phase - effective_age * env.reference_stride, 0
)
```

Use the same shape-safe expansion for `bootstrap_obs`; its new suffix is operational only at migration and has zero actor influence. Preserve all legacy bootstrap values exactly. Return one replaced immutable TrainState.

- [ ] **Step 5: Implement and validate the equivalence report**

The report must contain old/new input shapes, legacy parameter equality, legacy normalizer equality, zero new parameter/moment rows, finite new statistics, maximum action absolute/relative error, and `valid`. `validate_future_reference_migration_report` raises unless every predicate passes and both action errors are at most `1e-7`.

- [ ] **Step 6: Run focused GREEN and commit**

```bash
conda run -n diffsim python -m pytest \
  tests/test_shac_future_reference_migration.py \
  tests/test_shac_initialization.py -q
conda run -n diffsim ruff check \
  src/algorithms/shac/future_reference_migration.py \
  tests/test_shac_future_reference_migration.py
git add src/algorithms/shac/future_reference_migration.py \
  tests/test_shac_future_reference_migration.py
git commit -m "feat: migrate SHAC future reference checkpoints"
```

Expected: all selected tests and Ruff pass.

### Task 3: SHAC Resume Integration And Migration Evidence

**Files:**
- Modify: `src/algorithms/shac/algorithm.py:690-760,900-1170,1180-1295,2050-2220`
- Modify: `tests/test_shac_exact_resume.py`
- Modify: `tests/test_shac_cagrad_integration.py`

**Interfaces:**
- Adds train arguments `actor_reference_lookahead_steps: tuple[int, ...] = ()` and `allow_resume_actor_reference_lookahead_upgrade: bool = False`.
- Produces: `resolve_future_reference_resume_settings(...) -> tuple[tuple[int, ...], bool]`, where the boolean means migration is required.
- Writes: `migration_equivalence.json` before compilation and records the tuple in `hparams.json`.

- [ ] **Step 1: Write failing resume-resolution tests**

Cover four exact branches:

```python
self.assertEqual(
    resolve_future_reference_resume_settings(
        {"actor_reference_lookahead_steps": []},
        requested_steps=(4, 8, 12),
        allow_upgrade=True,
    ),
    ((4, 8, 12), True),
)
self.assertEqual(
    resolve_future_reference_resume_settings(
        {"actor_reference_lookahead_steps": [4, 8, 12]},
        requested_steps=(4, 8, 12),
        allow_upgrade=False,
    ),
    ((4, 8, 12), False),
)
```

Also require legacy exact resume with `()` to stay exact, and reject changed/removal offsets or upgrades without explicit authority.

- [ ] **Step 2: Run resume tests to establish RED**

```bash
conda run -n diffsim python -m pytest \
  tests/test_shac_exact_resume.py \
  tests/test_shac_cagrad_integration.py -q
```

Expected: new resolver and trainer arguments are absent.

- [ ] **Step 3: Wire the treatment through resume and environment creation**

Validate the requested tuple at the Python boundary. Resolve it before environment construction without allowing old hparams to overwrite an explicitly authorized append-only upgrade. Pass the tuple into G1 environment kwargs. After fresh expanded state initialization but before `select_initial_training_state`, call the migration only when the resolver returns `True`.

For treatment checkpoints whose saved tuple already matches, restore the complete saved state with no second migration. For all other mismatches, raise before JIT compilation.

- [ ] **Step 4: Persist evidence before compilation**

Write the validated report atomically to `save_dir/migration_equivalence.json` using sorted JSON. Add these hparams:

```python
"actor_reference_lookahead_steps": list(actor_reference_lookahead_steps),
"resume_future_reference_upgrade": bool(future_reference_upgrade),
"migration_equivalence_artifact": (
    "migration_equivalence.json" if future_reference_upgrade else None
),
```

Require the artifact for an upgrade before warmup. Do not write it on an unchanged exact resume.

- [ ] **Step 5: Run trainer integration GREEN and commit**

```bash
conda run -n diffsim python -m pytest \
  tests/test_shac_exact_resume.py \
  tests/test_shac_future_reference_migration.py \
  tests/test_shac_cagrad.py \
  tests/test_shac_cagrad_integration.py \
  tests/test_shac_microbatch.py -q
conda run -n diffsim ruff check src/algorithms/shac tests/test_shac_exact_resume.py
git add src/algorithms/shac/algorithm.py \
  tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py
git commit -m "feat: integrate future reference SHAC resume"
```

Expected: all selected tests and Ruff pass.

### Task 4: Immutable Runner And Replay-Free Evaluation

**Files:**
- Create: `tools/run_g1_cagrad_future_reference_continuation.py`
- Create: `tests/test_g1_cagrad_future_reference_runner.py`
- Modify: `tools/evaluate_g1_tracking.py:44-80,260-350`
- Modify: `tools/evaluate_g1_phase_grid.py:45-135`
- Modify: `tests/test_g1_tracking_evaluator.py`
- Modify: `tests/test_g1_phase_grid_evaluator.py`

**Interfaces:**
- Produces: `build_cagrad_future_reference_kwargs(profile_name, reference_path, seed, resume_from) -> dict`
- Adds evaluator CLI `--actor-reference-lookahead-steps 4 8 12`.
- Preserves explicit replay-free exact-phase evaluation.

- [ ] **Step 1: Write failing runner and evaluator tests**

The runner contract starts from `build_cagrad_continuation_kwargs`, uses exact E008 resume, and differs only by:

```python
{
    "actor_reference_lookahead_steps": (4, 8, 12),
    "allow_resume_actor_reference_lookahead_upgrade": True,
    "total_steps": 1_572_864,
}
```

Its existing `checkpoint_interval=196_608` yields the required midpoint/final checkpoints. Assert 64 H12 updates and reject every scientific CLI override. Evaluator tests must prove the flag constructs a 3280-value environment and the phase-grid child command includes the exact tuple.

- [ ] **Step 2: Run runner/evaluator tests to establish RED**

```bash
conda run -n diffsim python -m pytest \
  tests/test_g1_cagrad_future_reference_runner.py \
  tests/test_g1_tracking_evaluator.py \
  tests/test_g1_phase_grid_evaluator.py -q
```

Expected: missing runner module and evaluator arguments fail.

- [ ] **Step 3: Implement immutable runner**

Follow the existing CAGrad continuation runner's parser and `solver_context` main. Keep the lookahead tuple and upgrade authority inside the builder; expose no corresponding CLI option.

- [ ] **Step 4: Implement explicit evaluation wiring**

Add `actor_reference_lookahead_steps` to `make_evaluation_env` and pass a tuple from the parser. Add an optional tuple argument to `build_evaluator_command`; when nonempty, emit exactly one `--actor-reference-lookahead-steps` flag followed by its integer values. Existing commands remain byte-equivalent when the tuple is empty.

- [ ] **Step 5: Run focused GREEN, lint, compile, and commit**

```bash
conda run -n diffsim python -m pytest \
  tests/test_g1_cagrad_future_reference_runner.py \
  tests/test_g1_tracking_evaluator.py \
  tests/test_g1_phase_grid_evaluator.py \
  tests/test_g1_cagrad_continuation_runner.py \
  tests/test_g1_cagrad_h24_continuation_runner.py -q
conda run -n diffsim ruff check tools tests/test_g1_cagrad_future_reference_runner.py
conda run -n diffsim python -m py_compile \
  tools/run_g1_cagrad_future_reference_continuation.py \
  tools/evaluate_g1_tracking.py tools/evaluate_g1_phase_grid.py
git add tools/run_g1_cagrad_future_reference_continuation.py \
  tools/evaluate_g1_tracking.py tools/evaluate_g1_phase_grid.py \
  tests/test_g1_cagrad_future_reference_runner.py \
  tests/test_g1_tracking_evaluator.py tests/test_g1_phase_grid_evaluator.py
git commit -m "feat: add future reference CAGrad experiment"
```

Expected: all commands pass.

### Task 5: Exact-Checkpoint Preflight And Code Review

**Files:**
- Test only; repair only files owned by Tasks 1-4 if evidence fails.

**Interfaces:**
- Consumes exact E008 checkpoint and hparams from the canonical run directory.
- Produces a real `migration_equivalence.json` in a temporary smoke output and a clean executable commit.

- [ ] **Step 1: Run the focused scientific suite once**

```bash
JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest \
  tests/test_g1_tracking_environment.py \
  tests/test_g1_tracking_training_distribution.py \
  tests/test_shac_future_reference_migration.py \
  tests/test_shac_exact_resume.py \
  tests/test_shac_cagrad.py \
  tests/test_shac_cagrad_integration.py \
  tests/test_shac_microbatch.py \
  tests/test_g1_cagrad_future_reference_runner.py \
  tests/test_g1_tracking_evaluator.py \
  tests/test_g1_phase_grid_evaluator.py -q
```

Expected: all selected tests pass. Do not run the repository-wide suite unless a focused failure indicates a broader boundary.

- [ ] **Step 2: Run the exact E008 migration preflight without a logical update**

Load the exact checkpoint and hparams, build legacy and treatment environments under `g1-4x5`, initialize the expanded template, invoke the migration function, and validate its report. Assert the source SHA-256, source step, old/new input shapes `1540/3280`, maximum action errors at most `1e-7`, and no changed critic/non-actor leaves.

Expected: the report is valid and no GPU training update is consumed.

- [ ] **Step 3: Perform bounded code review and repair findings**

Review only representation ordering, history migration, optimizer-state preservation, resume authority, evaluator propagation, and default-off behavior. Any Critical or Important finding is fixed test-first and committed separately. Cosmetic findings do not trigger a heavy-suite rerun.

- [ ] **Step 4: Run final static checks and record executable commit**

```bash
conda run -n diffsim ruff check src tools tests
conda run -n diffsim python -m compileall -q src tools tests
git diff --check
git status --short
git rev-parse HEAD
```

Expected: lint, compilation, and diff checks pass; tracked worktree is clean; record the returned SHA for preregistration.

### Task 6: Preregister, Execute, Evaluate, And Curate E010

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260811-000.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generated: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/generated/`

**Interfaces:**
- Consumes: clean executable commit from Task 5 and exact E008 artifact identities.
- Produces: one guarded run under `runs/E-20260811-000/`, two replay-free phase grids, and curated registry evidence.

- [ ] **Step 1: Write the complete proposed experiment record**

Copy schema-v1 structure from E008/E009. Pin the clean code commit, empty dirty-patch SHA-256, conda environment, model/reference hashes, fixed 4x5 solver, exact E008 checkpoint and step, seed zero, GPU count one, and 120-minute wall-time. Use the immutable runner command and require:

```yaml
required_artifacts:
  - policy_final.pkl
  - hparams.json
  - log.npy
  - diag_log.json
  - migration_equivalence.json
  - checkpoint_step_1376256.pkl
  - checkpoint_step_1572864.pkl
```

Define outcomes `future-reference-advances`, `future-reference-finite-no-gain`, and `invalid-execution` with the frozen gate from Global Constraints.

- [ ] **Step 2: Validate, dry-run, and approve the bounded simulation**

```bash
cd /home/ubuntu/projects/diffsim2real-lab
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260811-000 --dry-run
```

Expected: validation and dry run pass. Set `approved_by: codex` and an exact UTC approval timestamp, revalidate, then commit only the registry record and generated projection.

- [ ] **Step 3: Launch the sole guarded training run**

```bash
uv run python -m tools.runexp E-20260811-000
```

Expected: return code zero within 120 minutes, exactly 64 actor updates, all required artifacts, valid migration report, finite 247-or-more numeric TrainState leaves, and finite valid five-bin CAGrad telemetry. Do not launch a successor while this run is active.

- [ ] **Step 4: Evaluate the two fixed checkpoints**

For each checkpoint, run the existing phase-grid evaluator with explicit `actor_reference_lookahead_steps=(4, 8, 12)` at phases `0,100,200,300,400`, nominal physics, no replay, and the fixed 4x5 solver. Produce each summary, five videos, and montage. Verify checkpoint/reference hashes before interpreting results.

- [ ] **Step 5: Select only a preregistered outcome and curate**

Select between midpoint and final by maximum minimum survival, then median, mean, and earliest checkpoint. Check the exact `19/48` aggregate gate and `23/50/72/16/40` phase floors. Update E010 result and `research/state/current.yaml` from deterministic artifacts, rebuild the vault, validate, run focused registry tests, and commit only owned canonical files.

```bash
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py build-vault
uv run pytest tests/test_research_registry.py -q
git diff --check
```

Expected: canonical evidence records what was learned, what remains unproven, and the next branch dictated by the registered outcome map.
