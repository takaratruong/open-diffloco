# Canonical SHAC G1 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a full-budget, upstream-faithful humanoid SHAC control and a corrected G1 solver pair with reference-relative zero initialization and real Open-DiffLoco-style domain randomization.

**Architecture:** Keep the existing SHAC optimizer and rollout as the authority. Add a focused G1 randomization module, opt-in reference-residual action semantics, and named solver profiles; expose them through a small canonical runner whose complete effective configuration is unit tested. The existing G1 modes remain byte-compatible by default.

**Tech Stack:** Python 3.11, JAX 0.4.38, MuJoCo/MJX 3.9.0, Flax 0.10.2, Optax 0.2.5, pytest, YAML experiment registry.

## Global Constraints

- Preserve all existing G1 behavior unless `reference_residual_control` or canonical domain randomization is explicitly enabled.
- Canonical learning uses 256 environments, horizon 12, 8,000,000 requested transitions, 16 critic updates, actor LR 5e-3, critic LR 5e-4, action noise 0.5 to 0.32, bootstrap scale 1.0, and no per-environment norm clipping.
- Finite-difference agreement is telemetry, never a rejection gate for a finite SHAC update.
- The upstream arm uses stock MJX 1/5; the G1 arm uses the same MJX equations with fixed-scan 4/5.
- Never bind any service to `0.0.0.0`.
- Training claims require deterministic replay-free checkpoint evaluation and video.

---

### Task 1: Reference-relative zero initialization

**Files:**
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `src/algorithms/shac/algorithm.py`
- Test: `tests/test_g1_tracking_environment.py`
- Test: `tests/test_actor_action_support.py`

**Interfaces:**
- Consumes: existing `G1TrackingEnv.step(state, action)` and `train(...)` environment construction.
- Produces: `reference_residual_control: bool`, `reference_residual_scale: float`, and `G1TrackingEnv.position_target(state, residual)`.

- [ ] **Step 1: Write the failing zero-head and residual-bound tests**

```python
def test_reference_residual_zero_is_phase_target(g1_env):
    state = g1_env.reset_at_phase(jax.random.PRNGKey(0), jp.array(0.0), jp.array(17))
    target = g1_env.position_target(state, jp.zeros(g1_env.action_dim))
    np.testing.assert_allclose(target, g1_env.qpos_reference[17, 7:], rtol=0, atol=0)


def test_reference_residual_uses_source_order_and_half_scale(g1_env):
    state = g1_env.reset_at_phase(jax.random.PRNGKey(0), jp.array(0.0), jp.array(17))
    residual = jp.ones(g1_env.action_dim)
    target = g1_env.position_target(state, residual)
    expected = g1_env.qpos_reference[17, 7:] + 0.5 * g1_env.action_scales
    np.testing.assert_allclose(target, expected, rtol=0, atol=1e-12)
```

- [ ] **Step 2: Run the tests and verify the new interface is absent**

Run: `pytest -q tests/test_g1_tracking_environment.py -k 'reference_residual'`

Expected: FAIL because `position_target` and constructor options do not exist.

- [ ] **Step 3: Implement opt-in reference-relative targets**

Add constructor fields and keep the direct-action default:

```python
self.reference_residual_control = bool(reference_residual_control)
self.reference_residual_scale = float(reference_residual_scale)
self.squash_actor_actions = self.reference_residual_control or self.clip_actions

def position_target(self, state: EnvState, action: jax.Array) -> jax.Array:
    prepared = self._prepare_action(action)
    if not self.reference_residual_control:
        return self.default_joints + prepared * self.action_scales
    phase = state.info["phase"]
    return (
        self.qpos_reference[phase, 7:]
        + self.reference_residual_scale * prepared * self.action_scales
    )
```

Use `position_target = self.position_target(state, action)` in `step`. Thread `reference_residual_control` and `reference_residual_scale` through `train` only for G1 variants.

- [ ] **Step 4: Prove legacy direct-action behavior remains exact**

Run: `pytest -q tests/test_actor_action_support.py tests/test_g1_tracking_environment.py -k 'action or reference_residual'`

Expected: all selected tests PASS; legacy direct targets retain their exact arrays.

- [ ] **Step 5: Commit**

```bash
git add src/envs/g1_tracking/environment.py src/algorithms/shac/algorithm.py tests/test_g1_tracking_environment.py tests/test_actor_action_support.py
git commit -m "feat: add phase-relative G1 residual control"
```

### Task 2: Per-environment G1 domain randomization

**Files:**
- Create: `src/envs/g1_tracking/randomization.py`
- Modify: `src/envs/g1_tracking/environment.py`
- Create: `tests/test_g1_tracking_randomization.py`

**Interfaces:**
- Produces: `sample_g1_randomization(key, difficulty, ranges) -> dict[str, jax.Array]` and `randomize_g1_model(base_model, base_arrays, info, pelvis_body_id) -> mjx.Model`.
- Consumes: sampled values stored under `friction_scale`, `mass_scale`, `kp_scale`, `kd_scale`, and `com_offset` in `EnvState.info`.

- [ ] **Step 1: Write pure sampling tests**

```python
def test_difficulty_zero_matches_upstream_randomization_semantics():
    values = sample_g1_randomization(jax.random.PRNGKey(1), jp.array(0.0), CANONICAL_RANGES)
    assert 0.5 <= float(values["friction_scale"]) <= 2.0
    assert 0.85 <= float(values["mass_scale"]) <= 1.15
    assert float(values["kp_scale"]) == 1.0
    assert float(values["kd_scale"]) == 1.0
    np.testing.assert_array_equal(values["com_offset"], np.zeros(3))


def test_difficulty_one_samples_complete_gain_and_com_ranges():
    samples = jax.vmap(lambda k: sample_g1_randomization(k, jp.array(1.0), CANONICAL_RANGES))(
        jax.random.split(jax.random.PRNGKey(2), 4096)
    )
    assert float(samples["kp_scale"].min()) >= 25.0 / 35.0
    assert float(samples["kp_scale"].max()) <= 45.0 / 35.0
    assert float(samples["kd_scale"].min()) >= 0.3 / 0.5
    assert float(samples["kd_scale"].max()) <= 0.7 / 0.5
    assert np.all(np.abs(np.asarray(samples["com_offset"])) <= np.array([0.05, 0.05, 0.04]))
```

- [ ] **Step 2: Run the sampling tests and verify failure**

Run: `pytest -q tests/test_g1_tracking_randomization.py`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the focused sampling module**

```python
@flax.struct.dataclass
class G1RandomizationRanges:
    friction: tuple[float, float]
    mass: tuple[float, float]
    kp_scale: tuple[float, float]
    kd_scale: tuple[float, float]
    com_offset: tuple[float, float, float]


def sample_g1_randomization(key, difficulty, ranges):
    k_friction, k_mass, k_kp, k_kd, k_com = jax.random.split(key, 5)
    kp_raw = jax.random.uniform(k_kp, (), minval=ranges.kp_scale[0], maxval=ranges.kp_scale[1])
    kd_raw = jax.random.uniform(k_kd, (), minval=ranges.kd_scale[0], maxval=ranges.kd_scale[1])
    com_raw = (2.0 * jax.random.uniform(k_com, (3,)) - 1.0) * jp.asarray(ranges.com_offset)
    return {
        "friction_scale": jax.random.uniform(k_friction, (), minval=ranges.friction[0], maxval=ranges.friction[1]),
        "mass_scale": jax.random.uniform(k_mass, (), minval=ranges.mass[0], maxval=ranges.mass[1]),
        "kp_scale": 1.0 + difficulty * (kp_raw - 1.0),
        "kd_scale": 1.0 + difficulty * (kd_raw - 1.0),
        "com_offset": difficulty * com_raw,
    }
```

- [ ] **Step 4: Write model-materialization tests**

```python
def test_randomized_model_scales_friction_mass_inertia_and_pelvis_ipos(g1_env):
    info = {"friction_scale": 1.5, "mass_scale": 1.1,
            "kp_scale": 0.8, "kd_scale": 1.2,
            "com_offset": jp.array([0.01, -0.02, 0.03])}
    model = g1_env._get_randomized_model(info)
    np.testing.assert_allclose(model.geom_friction, 1.5 * g1_env.base_friction)
    np.testing.assert_allclose(model.body_mass, 1.1 * g1_env.base_mass)
    np.testing.assert_allclose(model.body_inertia, 1.1 * g1_env.base_inertia)
    np.testing.assert_allclose(
        model.body_ipos[g1_env.pelvis_body_id],
        g1_env.base_ipos[g1_env.pelvis_body_id] + info["com_offset"],
    )
```

- [ ] **Step 5: Materialize the randomized model and gains in every G1 step**

Store immutable base arrays after `mjx.put_model`, implement `_get_randomized_model`, sample randomization only in `reset`, and change the physics loop to:

```python
model = self._get_randomized_model(state.info)
kp = self.kp * state.info["kp_scale"]
kd = self.kd * state.info["kd_scale"]

def physics_step(data, _):
    torque = jp.clip(kp * (position_target - data.qpos[7:]) - kd * data.qvel[6:],
                     -self.effort_limit, self.effort_limit)
    applied = jp.zeros(self.mj_model.nv).at[6:].set(torque)
    return mjx.step(model, data.replace(qfrc_applied=applied)), None
```

Use the same randomized model for `mjx.make_data`/`mjx.forward` during training resets. Keep `reset_at_phase` nominal for deterministic evaluation.

- [ ] **Step 6: Verify carried parameters and distinct vectorized models**

Run: `pytest -q tests/test_g1_tracking_randomization.py tests/test_g1_tracking_environment.py -k 'random or reset'`

Expected: PASS, including a two-environment test whose model arrays and one-step states differ.

- [ ] **Step 7: Commit**

```bash
git add src/envs/g1_tracking/randomization.py src/envs/g1_tracking/environment.py tests/test_g1_tracking_randomization.py
git commit -m "feat: randomize G1 physics per environment"
```

### Task 3: Canonical actor history and observation noise

**Files:**
- Modify: `src/envs/g1_tracking/environment.py`
- Test: `tests/test_g1_tracking_environment.py`

**Interfaces:**
- Produces: `actor_observation_noise: bool` and an exact 154-value per-frame noise mask tiled by `actor_history_len`.

- [ ] **Step 1: Write actor-noise mapping tests**

```python
def test_canonical_actor_noise_leaves_reference_and_actions_clean(g1_env_with_noise):
    mask = np.asarray(g1_env_with_noise.actor_noise_mask)
    assert mask.shape == (154,)
    np.testing.assert_array_equal(mask[:58], 0.0)
    np.testing.assert_array_equal(mask[58:64], 0.05)
    np.testing.assert_array_equal(mask[64:67], 0.2)
    np.testing.assert_array_equal(mask[67:96], 0.01)
    np.testing.assert_array_equal(mask[96:125], 0.01)
    np.testing.assert_array_equal(mask[125:154], 0.0)


def test_ten_frame_history_uses_shared_feature_normalizer(g1_env_history10):
    assert g1_env_history10.actor_obs_dim == 10 * g1_env_history10.actor_frame_obs_dim
```

- [ ] **Step 2: Run and observe the missing opt-in behavior**

Run: `pytest -q tests/test_g1_tracking_environment.py -k 'canonical_actor_noise or ten_frame_history'`

Expected: FAIL.

- [ ] **Step 3: Implement actor-only noise without changing critic observations**

```python
frame_noise = jp.concatenate((
    jp.zeros(58), jp.full(6, 0.05), jp.full(3, 0.2),
    jp.full(29, 0.01), jp.full(29, 0.01), jp.zeros(29),
))
self.actor_noise_mask = frame_noise

def _apply_obs_noise(self, obs, rng):
    if not self.actor_observation_noise:
        return obs
    mask = jp.tile(self.actor_noise_mask, self.actor_history_len)
    return obs + jax.random.uniform(rng, obs.shape, minval=-mask, maxval=mask)
```

- [ ] **Step 4: Verify complete observation behavior**

Run: `pytest -q tests/test_g1_tracking_environment.py -k 'obs or history or noise'`

Expected: PASS and existing noise-disabled tests remain exact.

- [ ] **Step 5: Commit**

```bash
git add src/envs/g1_tracking/environment.py tests/test_g1_tracking_environment.py
git commit -m "feat: add canonical G1 actor observation history"
```

### Task 4: Named solver profiles and stock/fixed identity

**Files:**
- Create: `src/envs/g1_tracking/solver_profiles.py`
- Modify: `tools/run_g1_tracking_rmr50_shac.py`
- Create: `tests/test_g1_solver_profiles.py`

**Interfaces:**
- Produces: `SOLVER_PROFILES`, `get_solver_profile(name)`, and `solver_context(profile)`.

- [ ] **Step 1: Write profile and identity tests**

```python
def test_registered_solver_profiles_are_exact():
    assert get_solver_profile("upstream-1x5") == SolverProfile(1, 5, False)
    assert get_solver_profile("g1-4x5") == SolverProfile(4, 5, True)
    assert get_solver_profile("diagnostic-10x20") == SolverProfile(10, 20, True)


def test_stock_and_fixed_one_iteration_match(g1_env_factory):
    stock = g1_env_factory(iterations=1, ls_iterations=5, fixed=False).step_from_fixture()
    fixed = g1_env_factory(iterations=1, ls_iterations=5, fixed=True).step_from_fixture()
    assert_tree_allclose(stock, fixed, rtol=0, atol=1e-12)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_g1_solver_profiles.py`

Expected: FAIL because the profile module is absent.

- [ ] **Step 3: Implement profiles and scoped contexts**

```python
@dataclass(frozen=True)
class SolverProfile:
    iterations: int
    ls_iterations: int
    fixed_scan: bool

SOLVER_PROFILES = {
    "upstream-1x5": SolverProfile(1, 5, False),
    "g1-4x5": SolverProfile(4, 5, True),
    "diagnostic-10x20": SolverProfile(10, 20, True),
}

def solver_context(profile):
    return fixed_mjx_solver_outer_loop() if profile.fixed_scan else nullcontext()
```

Thread iteration counts into the flexible `g1_tracking_rmr_50hz_source_step` environment rather than adding hard-coded environment subclasses.

- [ ] **Step 4: Run profile and existing solver tests**

Run: `pytest -q tests/test_g1_solver_profiles.py tests/test_g1_tracking_environment.py -k 'solver'`

Expected: PASS; context restoration is also tested after an injected exception.

- [ ] **Step 5: Commit**

```bash
git add src/envs/g1_tracking/solver_profiles.py tools/run_g1_tracking_rmr50_shac.py tests/test_g1_solver_profiles.py
git commit -m "feat: expose canonical G1 solver profiles"
```

### Task 5: Canonical G1 SHAC preset

**Files:**
- Create: `tools/run_canonical_g1_shac.py`
- Modify: `tools/run_g1_tracking_rmr50_shac.py`
- Modify: `src/algorithms/shac/algorithm.py`
- Create: `tests/test_canonical_g1_shac_runner.py`

**Interfaces:**
- Produces: `build_canonical_kwargs(profile_name, reference_path, seed) -> dict` and a CLI accepting only provenance/output fields plus `--solver-profile`.

- [ ] **Step 1: Write the complete preset contract test**

```python
def test_canonical_kwargs_match_open_diffloco_contract(reference_path):
    kw = build_canonical_kwargs("g1-4x5", reference_path, seed=42)
    expected = {
        "total_steps": 8_000_000, "num_envs": 256, "unroll_length": 12,
        "critic_iterations": 16, "actor_lr": 5e-3, "critic_lr": 5e-4,
        "action_noise_std_start": 0.5, "action_noise_std_end": 0.32,
        "actor_bootstrap_scale": 1.0, "actor_history_len": 10,
        "actor_hidden": (512, 256, 128), "actor_layer_norm": True,
        "actor_zero_output": True, "actor_per_env_grad_clip": None,
        "critic_per_env_grad_clip": None, "friction_range": (0.5, 2.0),
        "mass_range": (0.85, 1.15), "kp_range": (25.0, 45.0),
        "kd_range": (0.3, 0.7), "com_offset_range": (0.05, 0.05, 0.04),
        "push_velocity_range": (-1.0, 1.0), "push_interval_s": 4.0,
        "terrain": False, "reference_reset_noise_scale": 1.0,
        "reference_residual_control": True, "reference_residual_scale": 0.5,
        "actor_observation_noise": True, "solver_iterations": 4,
        "solver_ls_iterations": 5,
    }
    for key, value in expected.items():
        assert kw[key] == value
```

- [ ] **Step 2: Run and verify the canonical runner is absent**

Run: `pytest -q tests/test_canonical_g1_shac_runner.py`

Expected: FAIL on import.

- [ ] **Step 3: Implement the immutable preset**

Build the dictionary explicitly; do not derive scientific values from CLI defaults. Set checkpoint interval to `393_216` transitions so updates 128, 512, and 1,024 are archived and the final update 2,604 is always written. Set `env_variant="g1_tracking_rmr_50hz_source_step"`, `gradient_accumulation_steps=1`, `zero_difficulty_frac=0.0`, `curriculum_grace=800_000`, and `curriculum_steps=6_400_000`.

The CLI exposes `--solver-profile`, `--reference-path`, `--seed`, and `--output-root`; it does not expose scientific overrides.

- [ ] **Step 4: Preserve upstream gradient aggregation exactly**

For `actor_per_env_grad_clip is None`, retain:

```python
grads = jax.tree_util.tree_map(lambda g: jp.nanmean(g, axis=0), per_env_grads)
grads = jax.tree_util.tree_map(lambda g: jp.where(jp.isfinite(g), g, 0.0), grads)
```

Add truthful telemetry without changing `grads`; never apply a finite-difference gate.

- [ ] **Step 5: Verify preset, CLI rejection, and legacy runner behavior**

Run: `pytest -q tests/test_canonical_g1_shac_runner.py tests/test_g1_tracking_runner.py tests/test_shac_gradient_aggregation.py`

Expected: PASS; attempts to pass `--actor-lr` or other scientific overrides to the canonical CLI return code 2.

- [ ] **Step 6: Commit**

```bash
git add tools/run_canonical_g1_shac.py tools/run_g1_tracking_rmr50_shac.py src/algorithms/shac/algorithm.py tests/test_canonical_g1_shac_runner.py
git commit -m "feat: add immutable canonical G1 SHAC runner"
```

### Task 6: Differentiated two-environment smoke and evaluation receipts

**Files:**
- Create: `tools/smoke_canonical_g1_shac.py`
- Create: `tests/test_canonical_g1_shac_smoke.py`
- Modify: `tools/evaluate_g1_phase_grid.py`
- Test: `tests/test_g1_phase_grid_evaluator.py`

**Interfaces:**
- Produces: a JSON smoke receipt and evaluation summaries containing solver profile, effective randomization, initialization identity, finite fractions, and replay-free metrics.

- [ ] **Step 1: Write receipt-schema tests**

```python
def test_smoke_receipt_requires_real_randomization_and_update(receipt):
    assert receipt["num_envs"] == 2
    assert receipt["distinct_model_count"] == 2
    assert receipt["zero_head_reference_target_max_error"] <= 1e-12
    assert receipt["actor_grad_finite_fraction"] == 1.0
    assert receipt["critic_grad_finite_fraction"] == 1.0
    assert receipt["optimizer_update_norm"] > 0.0
```

- [ ] **Step 2: Run and verify receipt fields are missing**

Run: `pytest -q tests/test_canonical_g1_shac_smoke.py tests/test_g1_phase_grid_evaluator.py -k 'canonical or solver_profile'`

Expected: FAIL.

- [ ] **Step 3: Implement the bounded smoke**

Use two environments, one-step horizon, one actor update, and all 16 critic updates. Serialize sampled model hashes, zero-head target error, reward, actor/critic finite fractions, raw gradient norms, update norm, solver profile, runtime versions, and effective canonical hyperparameters. Return nonzero if any required value is absent or nonfinite.

- [ ] **Step 4: Extend evaluation provenance without changing physics**

Add the solver profile and checkpoint SHA-256 to phase-grid summary and video sidecars. Evaluation always uses nominal physical parameters, disabled action/observation noise, and exact phase resets.

- [ ] **Step 5: Run smoke/evaluation unit tests**

Run: `pytest -q tests/test_canonical_g1_shac_smoke.py tests/test_g1_phase_grid_evaluator.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/smoke_canonical_g1_shac.py tools/evaluate_g1_phase_grid.py tests/test_canonical_g1_shac_smoke.py tests/test_g1_phase_grid_evaluator.py
git commit -m "test: gate canonical G1 SHAC execution"
```

### Task 7: Register and launch the three full-budget arms

**Files:**
- Create in `diffsim2real-lab`: three `research/experiments/E-*.yaml` records
- Update in `diffsim2real-lab`: `research/state/current.yaml`

**Interfaces:**
- Consumes: clean implementation commit, passing smoke receipts, canonical runner, and frozen E016 humanoid port.
- Produces: Lane A upstream humanoid, Lane B stock-1/5 G1, and Lane B fixed-4/5 G1 training artifacts.

- [ ] **Step 1: Run focused and full preflight verification**

Run:

```bash
pytest -q tests/test_g1_tracking_randomization.py tests/test_g1_solver_profiles.py tests/test_canonical_g1_shac_runner.py tests/test_canonical_g1_shac_smoke.py
pytest -q
```

Expected: all tests PASS. Record exact counts and wall time.

- [ ] **Step 2: Run both solver smokes concurrently on separate physical GPUs**

Run the canonical smoke with `upstream-1x5` and `g1-4x5`. Expected: both return zero, publish complete receipts, and never bind a network service.

- [ ] **Step 3: Register immutable experiment contracts**

The records pin repository commit, environment lock, model/reference hashes, exact command, physical GPU, 8M transition budget, fixed checkpoints, no early termination for finite training, and replay-free evaluation rules. Lane A points at the frozen E016 worktree and its exact upstream hashes.

- [ ] **Step 4: Launch all three arms concurrently**

Use one physical GPU per arm. Lane A invokes the frozen upstream humanoid runner from initialization. The G1 commands differ only by `--solver-profile upstream-1x5` versus `--solver-profile g1-4x5`; seeds and all initial state hashes must match before the first update.

- [ ] **Step 5: Evaluate fixed checkpoints as they appear**

At actor updates 128, 512, 1,024, and 2,604, run deterministic replay-free phase-grid evaluation on GPUs not used by training. Evaluation must not pause or mutate training.

- [ ] **Step 6: Publish the solver decision**

Select by finite completion, phase-grid survival, body tracking, then cost. If equivalent, select stock 1/5; if fixed 4/5 materially improves behavior, select 4/5. Training reward and finite-difference exactness cannot decide.

- [ ] **Step 7: Commit registry and current-state updates**

```bash
git add research/experiments research/state/current.yaml docs/vault
git commit -m "research: compare canonical humanoid SHAC solvers"
```

## Completion gate

Implementation is complete only when the clean commit is pushed, focused and full tests pass, both solver smoke receipts pass, all three full-budget processes are running or complete under registered contracts, and the first immutable replay-free checkpoint comparison is published. A launched process without receipts is not completion.
