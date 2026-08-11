# G1 Fixed Torso-Assistance Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, fixed analytic torso-assistance curriculum to the exact E008 SHAC continuation while leaving the deployed actor and all zero-assistance behavior unchanged.

**Architecture:** Reuse one shared aligned-reference torso-wrench implementation in evaluation and training. Keep schedule calculation, zero-assistance sampling, validation, and resume resolution in a focused SHAC module; the main algorithm only injects the resulting per-environment wrench into actor rollouts and records telemetry. An immutable runner pins the 64-update E008 continuation.

**Tech Stack:** Python 3.10, JAX/MJX, MuJoCo, Flax, pytest, Ruff.

## Global Constraints

- Resume exact E008 at transition `1_327_104`; end at `1_720_320`.
- Use 256 physical environments, two accumulation shards, H12, and fixed `g1-4x5`.
- Linearly decay assistance from `1.0` at `1_327_104` to exact zero at `1_622_016`; keep the final 16 updates unassisted.
- Force 25 percent of environments to exact zero assistance throughout nonzero stages.
- Preserve all existing random streams by consuming the train step's currently unused sixth split key.
- Actor output remains 29-dimensional. Add no learned wrench head, scalar gate, observation, reward term, or deploy-time controller.
- The feature defaults off and the disabled actor-rollout path remains unchanged.
- Use test-driven development: every production behavior must first have an observed failing test.

---

### Task 1: Shared aligned torso-wrench computation

**Files:**
- Modify: `src/evaluation/g1_torso_wrench_oracle.py`
- Modify: `tools/evaluate_g1_frozen_torso_wrench_oracle.py`
- Modify: `tests/test_g1_torso_wrench_oracle.py`
- Modify: `tests/test_g1_frozen_torso_wrench_oracle.py`

**Interfaces:**
- Produces: `compute_environment_torso_wrench(environment, state, *, torso_slot, parameters, scale) -> tuple[jax.Array, jax.Array, jax.Array]`, returning the six-dimensional wrench plus actual torso linear and angular velocities.
- Consumes: existing `compute_torso_wrench`, environment `_body_state`, `_aligned_reference_body_targets`, and aligned reference velocity arrays.

- [ ] **Step 1: Write failing shared-path tests**

Add tests that compare `compute_environment_torso_wrench` against the evaluator's current `_wrench_for_state` on one real exact phase reset, verify returned shapes `(6,)`, `(3,)`, `(3,)`, and verify scale zero returns a bit-exact zero wrench.

```python
wrench, linear_velocity, angular_velocity = compute_environment_torso_wrench(
    env,
    state,
    torso_slot=7,
    parameters=parameters,
    scale=0.0,
)
np.testing.assert_array_equal(np.asarray(wrench), np.zeros(6))
assert linear_velocity.shape == (3,)
assert angular_velocity.shape == (3,)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n diffsim python -m pytest \
  tests/test_g1_torso_wrench_oracle.py \
  tests/test_g1_frozen_torso_wrench_oracle.py -q
```

Expected: fail because `compute_environment_torso_wrench` does not exist.

- [ ] **Step 3: Implement the shared helper**

Add the public function to `g1_torso_wrench_oracle.py`. It must reproduce the evaluator's current yaw-aligned position, quaternion, linear-velocity, and angular-velocity targets and call the existing bounded controller:

```python
def compute_environment_torso_wrench(
    environment: Any,
    state: Any,
    *,
    torso_slot: int,
    parameters: TorsoWrenchParameters,
    scale: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    positions, quaternions, linear_velocities, angular_velocities = (
        environment._body_state(state.data)
    )
    phase = state.info["phase"]
    reference_positions, reference_quaternions = (
        environment._aligned_reference_body_targets(
            positions[0], quaternions[0], phase
        )
    )
    reference_anchor = environment.body_quat_reference[phase, 0]
    yaw_delta = _yaw_quaternion(
        _quaternion_multiply(
            quaternions[0], _quaternion_conjugate(reference_anchor)
        )
    )
    reference_linear_velocity = _quaternion_apply(
        yaw_delta, environment.body_lin_vel_reference[phase, torso_slot]
    )
    reference_angular_velocity = _quaternion_apply(
        yaw_delta, environment.body_ang_vel_reference[phase, torso_slot]
    )
    wrench = compute_torso_wrench(
        parameters=parameters,
        actual_position=positions[torso_slot],
        actual_quaternion=quaternions[torso_slot],
        actual_linear_velocity=linear_velocities[torso_slot],
        actual_angular_velocity=angular_velocities[torso_slot],
        reference_position=reference_positions[torso_slot],
        reference_quaternion=reference_quaternions[torso_slot],
        reference_linear_velocity=reference_linear_velocity,
        reference_angular_velocity=reference_angular_velocity,
        scale=scale,
    )
    return wrench, linear_velocities[torso_slot], angular_velocities[torso_slot]
```

Use the module's quaternion helpers rather than importing private helpers from
the environment. Replace the evaluator's duplicated target calculation with
this function; retain inertial-COM velocity calculation and all artifact logic.

- [ ] **Step 4: Verify GREEN and unchanged evaluator contract**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/g1_torso_wrench_oracle.py \
  tools/evaluate_g1_frozen_torso_wrench_oracle.py \
  tests/test_g1_torso_wrench_oracle.py \
  tests/test_g1_frozen_torso_wrench_oracle.py
git commit -m "refactor: share aligned torso wrench computation"
```

### Task 2: Pure fixed-curriculum contract

**Files:**
- Create: `src/algorithms/shac/torso_wrench_curriculum.py`
- Create: `tests/test_shac_torso_wrench_curriculum.py`

**Interfaces:**
- Produces: `assistance_scale_at_step(step, *, start_step, end_step) -> jax.Array`.
- Produces: `sample_assistance_scales(key, *, num_envs, scheduled_scale, zero_fraction) -> jax.Array`.
- Produces: `validate_torso_wrench_assistance_configuration(...) -> None`.
- Produces: `resolve_torso_wrench_assistance_resume_settings(...) -> tuple[bool, int, int, float]`.

- [ ] **Step 1: Write failing schedule and validation tests**

Test exact endpoint values, midpoint `0.5`, post-end exact zero, deterministic
sampling, zero scheduled scale, invalid bounds/fractions, G1-only activation,
unauthorized resume changes, authorized activation from a legacy checkpoint,
and exact active-treatment restoration.

```python
assert float(assistance_scale_at_step(100, start_step=100, end_step=200)) == 1.0
assert float(assistance_scale_at_step(150, start_step=100, end_step=200)) == 0.5
assert float(assistance_scale_at_step(200, start_step=100, end_step=200)) == 0.0
assert float(assistance_scale_at_step(250, start_step=100, end_step=200)) == 0.0
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n diffsim python -m pytest \
  tests/test_shac_torso_wrench_curriculum.py -q
```

Expected: collection fails because the new module does not exist.

- [ ] **Step 3: Implement minimal pure helpers**

Use JAX operations for the schedule and sampling:

```python
def assistance_scale_at_step(step, *, start_step: int, end_step: int):
    progress = (
        (jp.asarray(step, dtype=jp.float32) - float(start_step))
        / float(end_step - start_step)
    )
    return jp.clip(1.0 - progress, 0.0, 1.0)


def sample_assistance_scales(
    key, *, num_envs: int, scheduled_scale, zero_fraction: float
):
    held_out = jax.random.uniform(key, (num_envs,)) < zero_fraction
    scale = jp.asarray(scheduled_scale, dtype=jp.float32)
    return jp.where(held_out, jp.zeros_like(scale), scale)
```

Validation must reject booleans as integers, nonfinite fractions, non-G1
activation, and `end_step <= start_step`. Resume resolution treats absent legacy
fields as disabled `(False, 0, 1, 0.0)` and requires `allow_change=True` for any
requested difference.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/algorithms/shac/torso_wrench_curriculum.py \
  tests/test_shac_torso_wrench_curriculum.py
git commit -m "feat: add fixed torso assistance schedule"
```

### Task 3: Opt-in SHAC rollout integration

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tests/test_shac_torso_wrench_curriculum.py`
- Modify: `tests/test_shac_exact_resume.py`
- Modify: `tests/test_shac_cagrad_integration.py`

**Interfaces:**
- Consumes Task 1's `compute_environment_torso_wrench` and existing `write_torso_wrench`.
- Consumes Task 2's schedule, sampling, validation, and resume functions.
- Adds `train` keyword arguments `torso_wrench_assistance`, `torso_wrench_assistance_start_step`, `torso_wrench_assistance_end_step`, `torso_wrench_assistance_zero_fraction`, and `allow_resume_torso_wrench_assistance_change`.

- [ ] **Step 1: Write failing integration contracts**

Assert defaults are disabled, fields are serialized to hparams, resume settings
are resolved before environment construction, the existing sixth PRNG split is
named `assistance_mask_key`, assistance scale is included in sharded
randomization, and enabled rollout code calls the shared helper then overwrites
only the torso row. Add a real helper-path gradient assertion:

```python
gradient = jax.grad(
    lambda scale: jp.sum(
        compute_environment_torso_wrench(
            env,
            state,
            torso_slot=7,
            parameters=parameters,
            scale=scale,
        )[0]
    )
)(jp.asarray(0.5))
assert np.isfinite(float(gradient))
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n diffsim python -m pytest \
  tests/test_shac_torso_wrench_curriculum.py \
  tests/test_shac_exact_resume.py \
  tests/test_shac_cagrad_integration.py -q
```

Expected: fail because `train` and its rollout do not expose assistance.

- [ ] **Step 3: Add validated opt-in configuration**

Add default-off arguments and documentation. Validate before loading or creating
the environment. Resolve resume treatment with the explicit allow-change flag.
After environment creation, resolve the named torso ID, its reference slot, and
physical parameters only when enabled. Store every field in `hparams.json`.

- [ ] **Step 4: Inject per-environment assistance into actor rollouts**

Keep the existing six-key split count and rename its unused last key. Compute
one scheduled scale and one `(effective_num_envs,)` sampled scale vector per
actor update. Append that vector to `all_randomization`, allowing existing
population sharding to preserve exact environment ownership.

Inside `actor_loss`, unpack the scalar scale and, only under the static enabled
branch, execute before `env.step`:

```python
torso_wrench, _, _ = compute_environment_torso_wrench(
    env,
    state,
    torso_slot=torso_slot,
    parameters=torso_wrench_parameters,
    scale=assistance_scale,
)
xfrc = write_torso_wrench(
    state.data.xfrc_applied,
    torso_body_id=torso_body_id,
    world_wrench=torso_wrench,
)
state = state.replace(data=state.data.replace(xfrc_applied=xfrc))
```

Add per-transition force norm, torque norm, and scalar scale. Aggregate current
scheduled scale, nonzero fraction, RMS force/torque, and maximum force/torque
into train metrics and checkpoint telemetry. Require all telemetry finite and
caps compliant when enabled.

- [ ] **Step 5: Verify GREEN and neighbors**

Run the Step 2 command, then:

```bash
conda run -n diffsim python -m pytest -q \
  tests/test_g1_torso_wrench_oracle.py \
  tests/test_g1_frozen_torso_wrench_oracle.py \
  tests/test_shac_residual_preview_adapter.py \
  tests/test_g1_frozen_residual_preview_runner.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/algorithms/shac/algorithm.py \
  tests/test_shac_torso_wrench_curriculum.py \
  tests/test_shac_exact_resume.py \
  tests/test_shac_cagrad_integration.py
git commit -m "feat: integrate torso assistance into SHAC rollouts"
```

### Task 4: Immutable E008 curriculum runner

**Files:**
- Create: `tools/run_g1_frozen_residual_assistance_curriculum.py`
- Create: `tests/test_g1_frozen_residual_assistance_runner.py`

**Interfaces:**
- Produces: `build_frozen_residual_assistance_kwargs(profile_name, reference_path, seed, resume_from) -> dict`.
- Produces: a CLI accepting only solver profile, reference path, seed, output root, and resume checkpoint.

- [ ] **Step 1: Write failing immutable-runner tests**

Compare the new builder with `build_frozen_residual_preview_kwargs`. Permit only
these changed keys:

```python
changed = {
    "total_steps",
    "torso_wrench_assistance",
    "torso_wrench_assistance_start_step",
    "torso_wrench_assistance_end_step",
    "torso_wrench_assistance_zero_fraction",
    "allow_resume_torso_wrench_assistance_change",
}
```

Assert endpoint `1_720_320`, start `1_327_104`, end `1_622_016`, zero fraction
`0.25`, checkpoint interval `49_152`, and rejection of CLI overrides such as
`--assistance-end-step`, `--zero-fraction`, `--num-envs`, and `--unroll-length`.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n diffsim python -m pytest \
  tests/test_g1_frozen_residual_assistance_runner.py -q
```

Expected: collection fails because the runner does not exist.

- [ ] **Step 3: Implement the immutable runner**

Build from `build_frozen_residual_preview_kwargs` and update only the six keys
listed above. Follow the existing solver-context, output-root, and JAX
configuration pattern used by `run_g1_frozen_residual_carried_continuation.py`.

- [ ] **Step 4: Verify GREEN and static quality**

Run:

```bash
conda run -n diffsim python -m pytest -q \
  tests/test_g1_frozen_residual_assistance_runner.py \
  tests/test_shac_torso_wrench_curriculum.py \
  tests/test_shac_exact_resume.py \
  tests/test_shac_cagrad_integration.py \
  tests/test_g1_torso_wrench_oracle.py \
  tests/test_g1_frozen_torso_wrench_oracle.py \
  tests/test_shac_residual_preview_adapter.py \
  tests/test_g1_frozen_residual_preview_runner.py
conda run -n diffsim ruff check \
  src/algorithms/shac/torso_wrench_curriculum.py \
  src/algorithms/shac/algorithm.py \
  src/evaluation/g1_torso_wrench_oracle.py \
  tools/evaluate_g1_frozen_torso_wrench_oracle.py \
  tools/run_g1_frozen_residual_assistance_curriculum.py \
  tests/test_shac_torso_wrench_curriculum.py \
  tests/test_g1_frozen_residual_assistance_runner.py
conda run -n diffsim python -m py_compile \
  src/algorithms/shac/torso_wrench_curriculum.py \
  src/algorithms/shac/algorithm.py \
  src/evaluation/g1_torso_wrench_oracle.py \
  tools/evaluate_g1_frozen_torso_wrench_oracle.py \
  tools/run_g1_frozen_residual_assistance_curriculum.py
git diff --check
```

Expected: all tests and static checks pass with a clean diff.

- [ ] **Step 5: Commit**

```bash
git add tools/run_g1_frozen_residual_assistance_curriculum.py \
  tests/test_g1_frozen_residual_assistance_runner.py
git commit -m "feat: add fixed assistance continuation runner"
```

### Task 5: Scientific review and registered execution

**Files:**
- Modify in the lab repository: `research/experiments/E-20260811-012.yaml`
- Modify in the lab repository: `research/state/current.yaml`
- Generate in the lab repository: corresponding Obsidian experiment, hypothesis, and current-state views.

**Interfaces:**
- Consumes: reviewed exact code SHA from Tasks 1-4 and exact E008 checkpoint/reference/model/controller hashes.
- Produces: one guarded GPU run with eight zero-wrench phase-grid checkpoint evaluations and a curated result.

- [ ] **Step 1: Review the exact implementation diff**

Verify default-off equivalence, resume determinism, wrench caps, exact-zero final
16 updates, unchanged 29-action export, and immutable runner delta. Repair only
findings demonstrated by tests, then rerun Task 4's verification commands.

- [ ] **Step 2: Register E012 before GPU execution**

Pin the exact code commit, dirty-patch hash, checkpoint/reference/model/controller
hashes, command, GPU, seed zero, 90-minute wall time, eight checkpoint artifacts,
and zero-wrench phase-grid evaluator. Register three outcomes: robust-key
advance, finite-no-gain, and invalid execution.

- [ ] **Step 3: Validate, launch, and monitor**

```bash
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py build-vault
uv run python tools/runexp.py E-20260811-012
```

Do not select by assisted reward. Evaluate every checkpoint with exact zero
wrench and curate the lexicographic `(minimum, median, mean)` comparison against
E008.

- [ ] **Step 4: Curate and commit evidence**

Update the experiment result, current state, hypothesis edge, artifact paths,
and generated vault views. Run registry validation, focused registry tests, and
`git diff --check`, then commit the bounded research result.
