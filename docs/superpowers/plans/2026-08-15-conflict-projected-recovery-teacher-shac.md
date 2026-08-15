# Conflict-Projected Recovery-Teacher SHAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, provenance-bound E036 teacher objective to E041's residual-adapter SHAC update, project away conflicts with the five-phase CAGrad direction, and run one guarded E042 treatment.

**Architecture:** A focused `recovery_teacher.py` module owns immutable dataset validation, the E038 imitation loss, resume authority, and tree-gradient projection/capping. `algorithm.train` loads the fixed teacher batch once, computes one full-batch teacher gradient per actor update, mixes it with the existing CAGrad direction, and persists complete telemetry. A guarded runner and a parameterized reuse of E041's paired evaluator/selector provide execution and scientific evidence without duplicating rollout machinery.

**Tech Stack:** Python 3.11, JAX, Flax, Optax, NumPy, MJX, pytest, Ruff.

## Global Constraints

- Teacher dataset is exactly `/home/ubuntu/projects/diffsim2real-lab/artifacts/E-20260815-036/e034_recovery_teacher_dataset.npz` at SHA-256 `203effe85e34794a76ebd344018e928f224d9cb8c9cedca9e2c4108f62343ad2`.
- Use exactly 416 transitions selected by the immutable E036 success mask.
- Teacher-gradient maximum norm ratio is exactly `0.5`; projection epsilon is exactly `1e-12`.
- Preserve E041's parent, hidden-feature/zero-head initialization, H24, effective population 512, five-bin CAGrad, carried resets, noise, anchor, physics, solver, seed, budget, and checkpoints.
- The feature is opt-in and every resume treatment change fails closed without explicit authority.
- No parameter sweep, new reset distribution, action-authority change, or teacher-data expansion.

---

### Task 1: Pure recovery-teacher contracts

**Files:**
- Create: `src/algorithms/shac/recovery_teacher.py`
- Create: `tests/test_shac_recovery_teacher.py`

**Interfaces:**
- Produces: `RecoveryTeacherBatch`, `load_recovery_teacher_batch(path, expected_sha256)`, `recovery_teacher_imitation_loss(...)`, `mix_conflict_projected_teacher_gradient(...)`, and `resolve_recovery_teacher_resume_settings(...)`.
- Consumes: existing `sha256_file`, JAX pytrees, NumPy NPZ evidence, and E036 tensor semantics.

- [ ] **Step 1: Write failing dataset and loss tests**

```python
def test_teacher_batch_selects_exact_success_rows_and_reproduces_e038_loss(tmp_path):
    batch = load_recovery_teacher_batch(tmp_path / "teacher.npz", expected_sha256=digest)
    assert batch.actor_obs.shape == (416, 3280)
    assert batch.parent_action.shape == (416, 29)
    loss = recovery_teacher_imitation_loss(
        batch.teacher_correction,
        batch.parent_action,
        batch.teacher_correction,
        batch.teacher_effective_action,
    )
    np.testing.assert_allclose(loss, 0.0, rtol=0, atol=1e-12)
```

Cover wrong SHA, missing/non-contiguous success mask, wrong shapes, nonfinite
values, and inconsistent clipped effective actions.

- [ ] **Step 2: Run the focused RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_recovery_teacher.py`

Expected: import failure for `src.algorithms.shac.recovery_teacher`.

- [ ] **Step 3: Implement immutable batch validation and loss**

```python
class RecoveryTeacherBatch(NamedTuple):
    actor_obs: np.ndarray
    parent_action: np.ndarray
    teacher_correction: np.ndarray
    teacher_effective_action: np.ndarray

def recovery_teacher_imitation_loss(predicted, parent, target, effective):
    correction = jnp.mean(jnp.square(predicted - target))
    predicted_effective = jnp.clip(parent + predicted, -1.0, 1.0)
    return correction + jnp.mean(jnp.square(predicted_effective - effective))
```

Validate the exact `(24,32,...)` source shapes, exact E034 success mask, 416
selected rows, finite tensors, and `clip(parent_action + correction) ==
effective_action` within `rtol=0, atol=1e-12`.

- [ ] **Step 4: Write failing projection and resume tests**

Test aligned, orthogonal, conflicting, zero-physics, ratio-cap, nonfinite, nested
pytree, missing resumed metadata, unauthorized changes, and explicit authority.

- [ ] **Step 5: Implement projection/capping and resume authority**

```python
dot = tree_dot(physics, teacher)
projected = tree_sub(teacher, tree_scale(physics, jnp.minimum(dot, 0.0) /
                                        (tree_squared_norm(physics) + 1e-12)))
scale = jnp.minimum(1.0, max_ratio * tree_norm(physics) /
                    (tree_norm(projected) + 1e-12))
combined = tree_add(physics, tree_scale(projected, scale))
```

Return the combined tree plus finite scalar telemetry. Require all four persisted
resume keys: enabled, resolved dataset path, dataset SHA-256, and ratio.

- [ ] **Step 6: Run GREEN and static checks**

Run the focused pytest, Ruff on the two files, `py_compile`, and `git diff --check`.

- [ ] **Step 7: Commit**

```bash
git add src/algorithms/shac/recovery_teacher.py tests/test_shac_recovery_teacher.py
git commit -m "feat: add conflict-projected recovery teacher"
```

### Task 2: Integrate teacher gradients into SHAC

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Create: `tests/test_shac_recovery_teacher_integration.py`
- Modify: `tests/test_shac_microbatch.py`

**Interfaces:**
- Consumes Task 1's batch, loss, gradient mixer, and resume resolver.
- Produces opt-in `train` arguments, hparams, checkpoint telemetry, and compiled actor-update behavior.

- [ ] **Step 1: Write failing API, disabled-equivalence, and enabled-update tests**

Require arguments:

```python
actor_recovery_teacher_dataset_path: str | None = None
actor_recovery_teacher_dataset_sha256: str | None = None
actor_recovery_teacher_gradient_ratio: float = 0.0
allow_resume_actor_recovery_teacher_change: bool = False
```

The disabled path must preserve the existing gradient/update exactly. The enabled
two-shard test must produce finite nonzero teacher and adapter gradients, exact
zero parent gradients/updates, and a teacher applied norm no greater than
`0.5 * physics_norm + 1e-7`.

- [ ] **Step 2: Run the integration RED**

Run the new integration tests plus `tests/test_shac_microbatch.py`; expect missing
arguments/telemetry failures while all legacy tests remain green.

- [ ] **Step 3: Add validation, loading, and normalization**

Resolve resume settings before environment compilation. Permit the treatment
only with residual preview adapter, frozen actor normalizer, five-bin CAGrad,
gradient accumulation two, and ratio exactly `0.5`. Load once and normalize the
416 stored histories with the frozen E023 normalizer, then select the newest
328-wide treatment frame.

- [ ] **Step 4: Add one teacher gradient per actor update**

```python
def teacher_loss_fn(params):
    _parent, adapter = split_residual_adapter_params(params)
    predicted = residual_preview_actor.apply(adapter, teacher_frames)
    return recovery_teacher_imitation_loss(
        predicted, teacher_parent_action, teacher_correction,
        teacher_effective_action,
    )

teacher_loss, teacher_grads = jax.value_and_grad(teacher_loss_fn)(state.actor_params)
mix = mix_conflict_projected_teacher_gradient(grads, teacher_grads, max_ratio=0.5)
grads = mix.combined_gradient
```

Run this after the existing five-bin CAGrad reduction and before the unchanged
adapter-only optimizer/update seam.

- [ ] **Step 5: Persist telemetry and hparams**

Add loss, raw/projected/applied norms, dot, cosine, applied scale, physics norm,
combined norm, and validity to normal diagnostics and every checkpoint row.
Persist the exact path/SHA/ratio/authority keys in `hparams.json`; reject a
checkpoint if enabled telemetry is absent, malformed, nonfinite, or violates the
norm cap.

- [ ] **Step 6: Run focused and neighboring GREEN suites**

Run recovery-teacher, microbatch, CAGrad, gradient aggregation, residual adapter,
and zero-head migration tests, then Ruff, `py_compile`, and diff check.

- [ ] **Step 7: Commit**

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_recovery_teacher_integration.py tests/test_shac_microbatch.py
git commit -m "feat: mix recovery teacher into SHAC"
```

### Task 3: Guarded E042 training runner

**Files:**
- Create: `tools/run_g1_conflict_projected_recovery_teacher.py`
- Create: `tests/test_g1_conflict_projected_recovery_teacher_runner.py`

**Interfaces:**
- Consumes E041's runner contracts and Task 2's train arguments/telemetry.
- Produces exact E042 preflight and `g1-conflict-projected-recovery-teacher-training-v1` validation.

- [ ] **Step 1: Write failing runner-delta and validation tests**

Assert that runner kwargs differ from E041 only by the four teacher treatment
keys. Require exact E036 hash, seed zero, clean code, E041 migration report,
eight checkpoint archives, complete finite teacher/CAGrad telemetry, exact
adapter-only updates, and final manifest-last validation.

- [ ] **Step 2: Run RED**

Run the new runner test and expect missing production module failure.

- [ ] **Step 3: Implement the guarded runner**

Reuse `build_zero_head_feature_transfer_kwargs`, E041 preflight, and E041 base
artifact validation. Add only the teacher dataset path/SHA, ratio `0.5`, and
explicit resume authority. Validate all checkpoint rows rather than only the
last diagnostic.

- [ ] **Step 4: Run GREEN and static checks**

Run the runner, E041 runner, teacher integration, and migration test modules;
then Ruff, `py_compile`, and diff check.

- [ ] **Step 5: Commit**

```bash
git add tools/run_g1_conflict_projected_recovery_teacher.py tests/test_g1_conflict_projected_recovery_teacher_runner.py
git commit -m "feat: add guarded recovery teacher run"
```

### Task 4: Reusable paired selection and scientific execution

**Files:**
- Modify: `tools/evaluate_g1_zero_head_feature_transfer.py`
- Create: `tools/select_g1_conflict_projected_recovery_teacher.py`
- Modify: `tests/test_g1_zero_head_feature_transfer_evaluator.py`
- Create: `tests/test_g1_conflict_projected_recovery_teacher_selector.py`
- Create after implementation: `research/experiments/E-20260815-042.yaml` in `/home/ubuntu/projects/diffsim2real-lab`

**Interfaces:**
- Consumes Task 3's training validation and E041's raw paired/ordinary evaluators.
- Produces `g1-conflict-projected-recovery-teacher-selection-v1` and the registered outcome labels.

- [ ] **Step 1: Write failing parameterized-selector tests**

Require E041 defaults to remain byte-equivalent. The E042 selector must require
its exact training protocol, use per-candidate paired parent survival, relabel
only the three valid scientific outcomes, retain all raw evidence hashes, and
fail closed on mixed protocols or commits.

- [ ] **Step 2: Implement the minimal selector parameterization**

Parameterize expected training protocol, output protocol, and outcome labels in
the aggregate function while retaining E041 defaults. The E042 CLI supplies:

```python
labels = {
    "zero-head-features-solve": "teacher-objective-solve",
    "zero-head-features-advance": "teacher-objective-advance",
    "zero-head-features-insufficient": "teacher-objective-insufficient",
}
```

- [ ] **Step 3: Run evaluator/selector GREEN and full static verification**

Run all E041/E042 evaluator tests, recovery-teacher tests, runner tests, Ruff,
`py_compile`, and diff check. Obtain one independent Critical/Important review
before registering GPU execution.

- [ ] **Step 4: Commit code and register E042**

Commit the selector, then register one exact seed-zero run with the reviewed
clean commit and immutable E023/E027/E036/E038 inputs. Validate the registry
before launch.

- [ ] **Step 5: Execute, evaluate, and curate**

Run the 64-update treatment. Evaluate updates 8/16/32/64 in parallel using the
paired-parent carried evaluator and ordinary grid. Run the manifest-last
selector, update the experiment result/current state/finding, rebuild the vault,
verify artifacts, commit only owned files, and push both repositories.
