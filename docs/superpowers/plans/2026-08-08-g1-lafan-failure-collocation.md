# G1 LAFAN Failure-Centered Collocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and smoke-test a 24-transition, two-step multiple-shooting transcription around the pinned LAFAN pelvis-height collapse without running an optimizer.

**Architecture:** A small scientific module owns fixed window dimensions, quaternion-aware physical-state defects, segment rollout, path constraints, and the scalar feasibility merit. The existing G1 environment exposes its already-validated PD/MJX physics advance so both strict evaluation and transcription call the same plant primitive. A CLI runs one complete physical derivative probe, then compiles one fixed-shape segment equality/JVP kernel and evaluates it across all 12 segments. It emits a fail-closed JSON report for dimensions, identity defects, contact activity, constraint values, directional derivatives, and the shapes of a thin framework-neutral episode mapping.

**Tech Stack:** Python 3.11, JAX float64, MuJoCo MJX 3.9, NumPy, `unittest`.

## Global Constraints

- Reference SHA-256 is `bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db`.
- The validated task uses 50 Hz control, reference stride 1, MJX solver iterations 4, and line-search iterations 5.
- Default failure window is phases 111 through 135, 24 transitions, 12 segments, and two actions per segment.
- Default dimensions are 1548 decision variables and 852 equality residuals.
- The strict uninterrupted evaluator remains the only behavioral decision gate.
- Do not add a policy, critic, optimizer framework, smoothing surrogate, or long solve.
- Corrected episode arrays match the frozen `sonic_grail_rollout_npz_v1` raw keys and chronology exactly.
- The smoke must exercise all 12 segments and 852 equality residuals, and it
  hard-fails unless at least one segment starts with active contact.

---

### Task 1: Fixed topology and quaternion-aware state defects

**Files:**
- Create: `src/envs/g1_tracking/failure_collocation.py`
- Create: `tests/test_g1_failure_collocation.py`

**Interfaces:**
- Produces: `FailureWindow`, `physical_state_defect(predicted_qpos, predicted_qvel, knot_qpos, knot_qvel)`, and `select_failure_window(arrays, window)`.
- `FailureWindow.decision_size` includes free state knots and every action; `equality_size` includes 70 defect coordinates and one quaternion norm residual per segment.

- [x] **Step 1: Write failing topology and identity-defect tests**

```python
def test_default_failure_window_has_fixed_small_dimensions(self):
    window = FailureWindow()
    self.assertEqual(window.knot_phases, tuple(range(111, 136, 2)))
    self.assertEqual(window.decision_size, 1548)
    self.assertEqual(window.equality_size, 852)

def test_state_defect_is_zero_for_equal_states_and_quaternion_sign(self):
    defect = physical_state_defect(qpos, qvel, -quaternion_only_sign(qpos), qvel)
    np.testing.assert_allclose(defect, 0.0, atol=1e-12)
```

- [x] **Step 2: Run RED**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_failure_collocation -v`

Expected: import failure because `failure_collocation.py` does not exist.

- [x] **Step 3: Implement the minimum topology, defect, and array selector**

```python
@dataclass(frozen=True)
class FailureWindow:
    start_phase: int = 111
    end_phase: int = 135
    segment_steps: int = 2

def physical_state_defect(predicted_qpos, predicted_qvel, knot_qpos, knot_qvel):
    # 3 root translation + 3 sign-canonical quaternion + 29 joints + 35 qvel.
    relative = quaternion_product(
        predicted_qpos[3:7], quaternion_inverse(knot_qpos[3:7])
    )
    relative = jnp.where(relative[0] < 0.0, -relative, relative)
    return jnp.concatenate(
        (
            predicted_qpos[:3] - knot_qpos[:3],
            2.0 * relative[1:],
            predicted_qpos[7:] - knot_qpos[7:],
            predicted_qvel - knot_qvel,
        )
    )
```

Validate exact shapes `(N,36)`, `(N,35)`, `(N,29)`, finite values, exact phase coverage, and normalized knot quaternions.

- [x] **Step 4: Run GREEN**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_failure_collocation -v`

Expected: all Task 1 tests pass.

### Task 2: Reuse the strict evaluator's physical plant

**Files:**
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `tests/test_g1_tracking_environment.py`

**Interfaces:**
- Produces: `G1TrackingEnv.advance_physics(data, action) -> tuple[mjx.Data, jax.Array, jax.Array]`.
- Returns advanced data, the prepared model-order action, and raw unclipped PD torque at the first physics substep for feasibility reporting.

- [x] **Step 1: Write a failing physical-parity test**

```python
def test_advance_physics_matches_nonterminal_step_data(self):
    state = env.reset_at_phase(key, jnp.array(0.0), jnp.array(0))
    data, prepared_action, raw_torque = env.advance_physics(state.data, action)
    stepped = env.step(state, action)
    np.testing.assert_allclose(data.qpos, stepped.data.qpos, atol=1e-12)
    np.testing.assert_allclose(data.qvel, stepped.data.qvel, atol=1e-12)
```

- [x] **Step 2: Run RED**

Run: `JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_environment.G1TrackingEnvironmentTest.test_advance_physics_matches_nonterminal_step_data -v`

Expected: `AttributeError` for missing `advance_physics`.

- [x] **Step 3: Extract the exact existing PD/MJX scan**

```python
def advance_physics(self, data, action):
    prepared = self._prepare_action(action)
    target = self.default_joints + prepared * self.action_scales
    initial_raw_torque = self.kp * (target - data.qpos[7:]) - self.kd * data.qvel[6:]
    # Execute the unchanged clipped-torque mjx.step scan for self.n_frames.
    return advanced, prepared, initial_raw_torque
```

Make `step()` delegate only its existing physics block to this method; do not change reset, reward, termination, observation, or metric behavior.

- [x] **Step 4: Run GREEN and focused regression tests**

Run: `JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_environment.G1TrackingEnvironmentTest.test_advance_physics_matches_nonterminal_step_data tests.test_g1_tracking_environment.G1TrackingEnvironmentTest.test_pd_target_and_torque_match_rmr_controller -v`

Expected: both tests pass.

### Task 3: Multiple-shooting equalities, constraints, and physical smoke

**Files:**
- Modify: `src/envs/g1_tracking/failure_collocation.py`
- Modify: `tests/test_g1_failure_collocation.py`
- Create: `tools/smoke_g1_failure_collocation.py`

**Interfaces:**
- Produces: `rollout_segment`, `multiple_shooting_equalities`, `failure_objective`, `physical_path_slacks`, `feasibility_merit`, and CLI JSON protocol `g1-lafan-failure-collocation-smoke-v1`.
- Consumes: `G1TrackingEnv.advance_physics`, fixed `FailureWindow`, and the pinned reference path.

- [x] **Step 1: Write failing mathematical identity and finite-gradient tests**

```python
def test_generated_segment_has_zero_multiple_shooting_defect(self):
    next_qpos, next_qvel = rollout_segment(step_fn, qpos, qvel, actions)
    residual = multiple_shooting_equalities(
        step_fn, qpos[None], qvel[None], next_qpos[None], next_qvel[None], actions
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)

def test_quadratic_merit_gradient_is_finite(self):
    decision = jnp.array([0.2, -0.3])
    value, gradient = jax.value_and_grad(
        lambda x: feasibility_merit(
            objective=jnp.sum(jnp.square(x)),
            equalities=x[:1],
            slacks=1.0 - jnp.abs(x),
        )
    )(decision)
    self.assertTrue(np.isfinite(value))
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(gradient)))
```

- [x] **Step 2: Run RED**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_failure_collocation -v`

Expected: imports fail for the new functions.

- [x] **Step 3: Implement only the scalar objective, explicit residuals, and slacks**

The objective is pelvis-height tracking plus `1e-3` joint tracking, actor-action deviation, and action-rate cost. Equality residuals concatenate every 70-coordinate segment defect with free-knot quaternion norm. Differentiable inequality slacks concatenate the four exact RMR hard limits, action support, and raw torque authority. Contact penetration is returned as a separate value diagnostic because its active-set JVP was non-finite in the first physical probe; preserve and report that failure without smoothing. `feasibility_merit` adds squared equality residuals and squared negative-slack violations; it does not solve.

- [x] **Step 4: Run GREEN unit tests**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_failure_collocation -v`

Expected: all failure-collocation unit tests pass.

- [ ] **Step 5: Implement and run the one-segment physical smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ubuntu/miniconda3/envs/rl/bin/python -u \
tools/smoke_g1_failure_collocation.py \
  --reference-path /home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/artifacts/E-20260808-000/reference/dance1_subject2_f122_422_50hz.npz \
  --reference-sha256 bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db \
  --checkpoint-path /home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/artifacts/E-20260808-012/carried_run_root/training_runs/shac_20260807_234306/checkpoint_step_688128.pkl \
  --config-path /home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/artifacts/E-20260808-012/carried_run_root/training_runs/shac_20260807_234306/hparams.json \
  --grail-commit 1e0d9df6e62f535387e0e7c37a1943fe8f4465a4 \
  --code-commit COMMITTED_IMPLEMENTATION_SHA \
  --output artifacts/g1_lafan_failure_collocation_smoke.json
```

Expected: exit zero; report decision/equality dimensions 1548/852,
near-zero one-segment identity equalities, all 852 finite full-window
equalities and their finite JVP, at least one active-contact segment, finite
objective gradient, finite constraint JVP, solver 4/5, float64 true, and exact
reference/checkpoint/config/model/GRAIL/code identities.

Observed: all 12 physical segment equality/JVP probes were finite and the
active-contact assertion passed, but the CLI failed closed before writing the
artifact because the terminal constraint JVP was non-finite. A separate direct
probe isolated only `anchor_xy_error`: at zero error its `norm` JVP was NaN.
The exact feasible-set squared slack rewrite was then preregistered,
implemented, and covered by algebraic-equivalence and boundary-sign tests. No
solve is permitted until its bounded physical rerun passes; contact remains a
value-only diagnostic.

Final result: terminal/action/torque/contact directional derivatives were all
finite in the pre-gate diagnostic, with 10 active-contact segments and all 852
full-window equality JVP entries finite. The artifact is not passing because
its one-segment identity maximum was `0.019016480604575836`. After restoring
direct segment assembly and adding a hard `1e-8` identity check, the final run
failed closed at that check. Do not run a solve or treat the ignored diagnostic
JSON as a completed smoke artifact.

Follow-up boundary result: a `custom_jvp` wrapper with the exact direct plant
as primal and ordinary JAX JVP as tangent passed the contact-free CPU MJX
microcase (bitwise primal equality and centered-FD tangent agreement). The real
G1 rerun still failed the hard `1e-8` identity gate. Conclude this MJX
collocation path is blocked; do not add more approximation layers or run an
optimizer.

- [ ] **Step 6: Run focused verification and commit**

Run:

```bash
JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_failure_collocation \
  tests.test_g1_collocation_transfer \
  tests.test_g1_tracking_environment.G1TrackingEnvironmentTest.test_advance_physics_matches_nonterminal_step_data \
  -v
git diff --check
```

Expected: all selected tests pass and `git diff --check` emits no output.

### Task 4: Thin framework-neutral corrected-episode mapping

**Files:**
- Modify: `src/envs/g1_tracking/failure_collocation.py`
- Modify: `tests/test_g1_failure_collocation.py`
- Modify: `tools/smoke_g1_failure_collocation.py`

**Interfaces:**
- Produces: `corrected_episode_mapping(...) -> dict[str, object]` and
  `world_body_kinematics(...)`.
- Holds exact canonical raw arrays `root_pos(T,3)`, `root_rot(T,4)`,
  `root_ang_vel(T,3)`, `body_pos(T,30,3)`, `body_rot(T,30,4)`,
  `body_lin_vel(T,30,3)`, `joint_pos(T,29)`, `joint_vel(T,29)`,
  `action(T,29)`, ordered names/defaults/scales, timing/semantics, and exact
  DiffSim provenance.
- Keeps `state_t/action_t` raw chronology; it does not apply or store TML's
  P2/+1 interpretation.

- [x] **Step 1: Write a failing mapping test**

```python
def test_corrected_episode_mapping_exposes_50hz_state_action_contract(self):
    episode = corrected_episode_mapping(
        phases=np.arange(111, 124),
        qpos=np.zeros((13, 36)),
        qvel=np.zeros((13, 35)),
        body_state=self.body_state,
        actions=np.zeros((13, 29)),
        metadata=self.metadata,
    )
    self.assertEqual(episode["schema_version"], "sonic_grail_rollout_npz_v1")
    self.assertEqual(episode["action"].shape, (13, 29))
    self.assertEqual(episode["joint_names"], self.joint_names)
```

- [x] **Step 2: Run RED**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_failure_collocation -v`

Expected: import failure because `corrected_episode_mapping` does not exist.

- [x] **Step 3: Implement only the local mapping**

Require state/action count `T >= 13`, exact dimensions 36/35/29, all 30
non-world body states, finite arrays, WXYZ quaternions, exact canonical names,
defaults, scales, timing/semantics, standard provenance, and all seven required
DiffSim correction provenance values. Reorder by name, convert actions while
preserving their physical PD target, and return copies in a plain dictionary.

- [x] **Step 4: Run GREEN and report the identity mapping in the smoke**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_failure_collocation -v`

Expected: all mapping tests pass; the physical smoke reports mapping keys and
shapes with correction method `identity-smoke` but publishes no competing
episode file.
