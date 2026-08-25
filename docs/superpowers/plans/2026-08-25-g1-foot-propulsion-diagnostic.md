# G1 Foot-Propulsion Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a replay-free diagnostic that measures whether foot contact supplies the forward force required by the reference and publishes the result beside the rollout.

**Architecture:** A pure evaluation module owns coordinate transforms, reference finite differences, validation, and summary calculations. The existing G1 evaluator captures simulator state immediately after each completed step, writes aligned arrays into its canonical NPZ, and renders one diagnostic PNG without changing training behavior.

**Tech Stack:** Python 3.11, JAX, NumPy, Matplotlib, MJX, pytest.

## Global Constraints

- Do not modify training, reward, observations, contact parameters, or E005.
- Treat `data.qfrc_constraint[:3]` as the net floating-base constraint force.
- Express forward force in the current pelvis-yaw frame.
- Preserve one output row per evaluator record and reject non-finite evidence.
- Use a one-sided reference-velocity finite difference at the last frame.

---

### Task 1: Pure propulsion calculations

**Files:**
- Create: `src/evaluation/g1_foot_propulsion.py`
- Create: `tests/test_g1_foot_propulsion.py`

**Interfaces:**
- Produces: `yaw_frame_vector(world_vector, root_quaternion) -> jax.Array`
- Produces: `constraint_propulsion_sample(*, qfrc_constraint, root_quaternion, dt) -> tuple[jax.Array, jax.Array]`
- Produces: `reference_required_force(*, reference_root_velocity, phase, stride, dt, total_mass, root_quaternion) -> jax.Array`
- Produces: `summarize_propulsion(actual_forward, required_forward) -> dict[str, float]`

- [ ] **Step 1: Write failing transform and force tests**

```python
def test_yaw_frame_vector_rotates_world_force_into_heading_frame():
    quaternion = yaw_quaternion(np.pi / 2)
    actual = yaw_frame_vector(np.array([0.0, 10.0, 3.0]), quaternion)
    np.testing.assert_allclose(actual, [10.0, 0.0, 3.0], atol=1e-6)

def test_constraint_sample_returns_force_and_interval_impulse():
    force, impulse = constraint_propulsion_sample(
        qfrc_constraint=np.array([4.0, -2.0, 10.0, 99.0]),
        root_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        dt=0.04,
    )
    np.testing.assert_array_equal(force, [4.0, -2.0, 10.0])
    np.testing.assert_array_equal(impulse, np.array([4.0, -2.0, 10.0]) * 0.04)
```

- [ ] **Step 2: Run Task 1 tests and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_foot_propulsion.py`

Expected: collection fails because `src.evaluation.g1_foot_propulsion` does not exist.

- [ ] **Step 3: Implement the minimal pure module**

Use normalized quaternion yaw to construct world-heading basis vectors, slice exactly the first three generalized constraint-force entries, multiply by a positive finite `dt`, and compute the reference acceleration from `velocity[min(phase + stride, last)] - velocity[phase]`. Validate all shapes and finite values before returning arrays.

- [ ] **Step 4: Add endpoint, zero, invalid-input, and summary tests**

```python
def test_reference_endpoint_uses_zero_one_sided_acceleration():
    required = reference_required_force(
        reference_root_velocity=np.array([[1.0, 0.0, 0.0]]),
        phase=0,
        stride=1,
        dt=0.04,
        total_mass=33.0,
        root_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(required, np.zeros(3))
```

- [ ] **Step 5: Run Task 1 tests and static checks**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_foot_propulsion.py`

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check src/evaluation/g1_foot_propulsion.py tests/test_g1_foot_propulsion.py`

Expected: all pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/evaluation/g1_foot_propulsion.py tests/test_g1_foot_propulsion.py
git commit -m "feat: compute G1 foot propulsion diagnostics"
```

### Task 2: Evaluator artifacts and E005 diagnostic

**Files:**
- Modify: `tools/evaluate_g1_tracking.py`
- Modify: `tests/test_g1_tracking_evaluator.py`

**Interfaces:**
- Consumes: all Task 1 functions.
- Produces: aligned `constraint_force_world`, `constraint_force_yaw`, `constraint_impulse_yaw`, `foot_support`, `reference_required_force_yaw`, `torso_pitch`, and `applied_torso_force` arrays in `evaluation.npz`.
- Produces: `foot_propulsion_diagnostics.png` and JSON scalar keys `propulsion_forward_error_rms` and `propulsion_forward_force_peak_abs`.

- [ ] **Step 1: Write failing evaluator artifact tests**

Extend the evaluator fixture assertion to require the seven NPZ arrays, equal leading dimensions, finite values, both summary keys, and a nonempty PNG. Add a pure plotting test that rejects unequal row counts rather than silently truncating.

- [ ] **Step 2: Run focused evaluator tests and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_tracking_evaluator.py -k propulsion`

Expected: failure because the new evidence keys and PNG do not exist.

- [ ] **Step 3: Capture one propulsion sample per completed step**

Immediately after `compiled_step`, compute the constraint sample from the returned state, support with `env.foot_support_signature(state.data)`, reference requirement from `env.body_lin_vel_reference[:, 0]`, torso pitch from `state.data.xquat[env.body_ids[7]]`, and torso assistance from `state.data.xfrc_applied[env.body_ids[7], :3]`. Append before the matching `records.append` call.

- [ ] **Step 4: Publish NPZ, JSON, and plot**

Stack the arrays only after asserting `len(array) == len(records)`. Write them through the existing atomic evaluation path, call a focused Matplotlib plot helper, and merge `summarize_propulsion` into the summary.

- [ ] **Step 5: Run focused and neighboring verification**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_foot_propulsion.py tests/test_g1_tracking_evaluator.py`

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check src/evaluation/g1_foot_propulsion.py tools/evaluate_g1_tracking.py tests/test_g1_foot_propulsion.py tests/test_g1_tracking_evaluator.py`

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile src/evaluation/g1_foot_propulsion.py tools/evaluate_g1_tracking.py`

Run: `git diff --check`

Expected: all pass.

- [ ] **Step 6: Run the exact E005 update-128 replay-free evaluation**

Use the same evaluator command and pinned inputs recorded by the completed update-128 milestone, changing only the output directory to an immutable diagnostic child. Require `learned_torso_wrench` and `applied_torso_force` to be bit-exact zero before interpreting the plot.

- [ ] **Step 7: Commit Task 2**

```bash
git add tools/evaluate_g1_tracking.py tests/test_g1_tracking_evaluator.py
git commit -m "feat: publish G1 foot propulsion diagnostics"
```
