# G1 Frozen Preview Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train only the current-frame future-reference input columns of the migrated E008 actor, prove that every parent-policy value remains frozen, and determine whether differentiable loss improvements predict replay-free survival.

**Architecture:** Reuse the validated `(4, 8, 12)` future-reference migration, but apply an exact PyTree mask after five-bin CAGrad so that only 174 rows of `Dense_0/kernel` for the newest history frame can change. A focused helper owns mask construction, Adam-state preservation, parent-action reconstruction, and telemetry; the SHAC trainer only wires that helper into optimization, normalizer freezing, logging, and resume metadata. An immutable Stage-1 runner resumes E008 for 64 H12 updates, and a pure audit joins midpoint/final training-bin losses to existing replay-free phase-grid results.

**Tech Stack:** Python 3.11, JAX, Flax parameter PyTrees, Optax chained clip/Adam state, NumPy, pytest, Ruff, pickle/JSON experiment artifacts.

## Global Constraints

- Resume exact E008 at transition `1_179_648`; do not resume E010.
- Use preview offsets `(4, 8, 12)`, history length `10`, legacy frame width `154`, treatment frame width `328`, and first hidden width `512`.
- Exactly `174 * 512 = 89_088` actor parameters may train: the newest frame's preview suffix in `params/Dense_0/kernel`.
- Freeze all other actor parameters, all corresponding Adam `mu`/`nu` entries, and the complete actor normalizer; advance the shared Adam count and authorized moments.
- Keep E008's H12, 256 physical environments, accumulation factor 2, five CAGrad bins, 4x5 solver, seed 0, and 393,216-transition continuation budget.
- No PPO teacher, replay controller, hard-coded phase switch, preview-offset sweep, learning-rate sweep, horizon sweep, seed sweep, or hardware claim.
- Stage 2 is conditional and receives a separate plan only after Stage-1 evidence; this plan implements Stage 1 and the objective-survival audit only.

---

## File Map

- Create `src/algorithms/shac/preview_adapter.py`: pure mask, masked-Adam, parent-observation, phase telemetry, and frozen-state audit functions.
- Create `tests/test_shac_preview_adapter.py`: numerical contracts for the pure helper.
- Modify `src/algorithms/shac/algorithm.py`: configuration/resume validation, actor rollout telemetry, masked optimization, normalizer freezing, checkpoint metrics, and hparams.
- Modify `tests/test_shac_exact_resume.py`: exact-resume and legacy-treatment validation for the new flag.
- Modify `tests/test_shac_cagrad_integration.py`: trainer-level preview-adapter wiring and diagnostics.
- Create `tools/run_g1_frozen_preview_adapter_continuation.py`: immutable E008 Stage-1 runner.
- Create `tests/test_g1_frozen_preview_adapter_runner.py`: exact runner delta and CLI contract.
- Create `tools/evaluate_preview_objective_survival.py`: pure join and rank-agreement audit.
- Create `tests/test_preview_objective_survival_evaluator.py`: deterministic audit math and provenance rejection.
- Create `docs/vault/experiments/E-20260811-001-g1-frozen-preview-adapter.md`: preregistration and eventual result card.
- Modify `experiments/registry.yaml`: registered Stage-1 command, artifact contract, and result status.

### Task 1: Exact Preview Mask and Adam Preservation

**Files:**
- Create: `src/algorithms/shac/preview_adapter.py`
- Create: `tests/test_shac_preview_adapter.py`

**Interfaces:**
- Consumes: migrated Flax actor parameters and Optax state from `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))`.
- Produces: `build_current_preview_mask(params, *, history_len, legacy_frame_dim, treatment_frame_dim)`, `apply_preview_adapter_update(optimizer, gradients, optimizer_state, params, mask)`, `masked_tree_l2_norm(tree, mask)`, `max_abs_outside_mask(tree, mask)`, and `frozen_preview_state_drift(parent_params, candidate_params, parent_optimizer_state, candidate_optimizer_state, parent_normalizer, candidate_normalizer, mask)`.

- [ ] **Step 1: Write failing mask and inherited-moment tests**

```python
def test_mask_selects_only_newest_preview_rows():
    params = {"params": {"Dense_0": {"kernel": jp.zeros((3280, 512)), "bias": jp.zeros(512)}, "Dense_1": {"kernel": jp.zeros((512, 29))}}}
    mask = build_current_preview_mask(
        params, history_len=10, legacy_frame_dim=154, treatment_frame_dim=328
    )
    kernel = mask["params"]["Dense_0"]["kernel"].reshape(10, 328, 512)
    assert int(jp.sum(kernel)) == 174 * 512
    np.testing.assert_array_equal(kernel[:9], False)
    np.testing.assert_array_equal(kernel[9, :154], False)
    np.testing.assert_array_equal(kernel[9, 154:], True)
    assert not bool(jp.any(mask["params"]["Dense_0"]["bias"]))
    assert not bool(jp.any(mask["params"]["Dense_1"]["kernel"]))


def test_masked_adam_preserves_frozen_values_and_nonzero_inherited_moments():
    params = {"params": {"Dense_0": {"kernel": jp.arange(24.0).reshape(6, 4)}}}
    mask = build_current_preview_mask(
        params, history_len=2, legacy_frame_dim=2, treatment_frame_dim=3
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-2))
    state = optimizer.init(params)
    _, state = optimizer.update(jax.tree.map(jp.ones_like, params), state, params)
    original_params = jax.tree.map(jp.array, params)
    original_mu = jax.tree.map(jp.array, state[1][0].mu)
    for _ in range(3):
        updates, state, diagnostics = apply_preview_adapter_update(
            optimizer, jax.tree.map(lambda x: jp.full_like(x, 2.0), params), state, params, mask
        )
        params = optax.apply_updates(params, updates)
        assert float(diagnostics["frozen_update_max_abs"]) == 0.0
        assert float(diagnostics["frozen_moment_drift_max_abs"]) == 0.0
    frozen = jax.tree.map(lambda x, m: jp.where(m, 0.0, x), params, mask)
    expected = jax.tree.map(lambda x, m: jp.where(m, 0.0, x), original_params, mask)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(frozen), jax.tree.leaves(expected), strict=True))
    frozen_mu = jax.tree.map(lambda x, m: jp.where(m, 0.0, x), state[1][0].mu, mask)
    expected_mu = jax.tree.map(lambda x, m: jp.where(m, 0.0, x), original_mu, mask)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(frozen_mu), jax.tree.leaves(expected_mu), strict=True))
    assert int(state[1][0].count) > 1


def test_frozen_state_audit_ignores_authorized_rows_but_detects_every_frozen_leaf():
    report = frozen_preview_state_drift(
        parent_params, candidate_params, parent_opt, candidate_opt,
        parent_normalizer, candidate_normalizer, mask,
    )
    assert report == {
        "frozen_parameter_max_abs": 0.0,
        "frozen_mu_max_abs": 0.0,
        "frozen_nu_max_abs": 0.0,
        "actor_normalizer_max_abs": 0.0,
        "valid": True,
    }
```

- [ ] **Step 2: Run RED**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_preview_adapter.py -q`

Expected: collection failure because `src.algorithms.shac.preview_adapter` does not exist.

- [ ] **Step 3: Implement exact mask and chained-Adam update**

```python
def build_current_preview_mask(params, *, history_len, legacy_frame_dim, treatment_frame_dim):
    if history_len < 1 or not 0 < legacy_frame_dim < treatment_frame_dim:
        raise ValueError("preview dimensions must describe an append-only history")
    mask = jax.tree.map(lambda value: jp.zeros(value.shape, dtype=bool), params)
    kernel = params["params"]["Dense_0"]["kernel"]
    expected_rows = history_len * treatment_frame_dim
    if kernel.ndim != 2 or kernel.shape[0] != expected_rows:
        raise ValueError("Dense_0 kernel does not match the preview history layout")
    kernel_mask = jp.zeros(kernel.shape, dtype=bool).reshape(
        history_len, treatment_frame_dim, kernel.shape[1]
    )
    kernel_mask = kernel_mask.at[-1, legacy_frame_dim:, :].set(True).reshape(kernel.shape)
    mutable = flax.core.unfreeze(mask) if isinstance(mask, flax.core.FrozenDict) else dict(mask)
    mutable["params"] = dict(mutable["params"])
    mutable["params"]["Dense_0"] = dict(mutable["params"]["Dense_0"])
    mutable["params"]["Dense_0"]["kernel"] = kernel_mask
    return flax.core.freeze(mutable) if isinstance(params, flax.core.FrozenDict) else mutable


def apply_preview_adapter_update(optimizer, gradients, optimizer_state, params, mask):
    if len(optimizer_state) != 2 or len(optimizer_state[1]) != 2 or not isinstance(optimizer_state[1][0], optax.ScaleByAdamState):
        raise ValueError("preview adapter requires clip_by_global_norm followed by Adam")
    masked_gradients = jax.tree.map(lambda g, m: jp.where(m, g, 0.0), gradients, mask)
    proposed_updates, proposed_state = optimizer.update(masked_gradients, optimizer_state, params)
    updates = jax.tree.map(lambda u, m: jp.where(m, u, 0.0), proposed_updates, mask)
    old_adam = optimizer_state[1][0]
    new_adam = proposed_state[1][0]
    merged_adam = new_adam._replace(
        mu=jax.tree.map(lambda new, old, m: jp.where(m, new, old), new_adam.mu, old_adam.mu, mask),
        nu=jax.tree.map(lambda new, old, m: jp.where(m, new, old), new_adam.nu, old_adam.nu, mask),
    )
    merged_state = (proposed_state[0], (merged_adam, proposed_state[1][1]))
    diagnostics = {
        "preview_gradient_norm": masked_tree_l2_norm(gradients, mask),
        "preview_update_norm": masked_tree_l2_norm(updates, mask),
        "frozen_update_max_abs": max_abs_outside_mask(updates, mask),
        "frozen_moment_drift_max_abs": jp.maximum(
            max_abs_outside_mask(jax.tree.map(jp.subtract, merged_adam.mu, old_adam.mu), mask),
            max_abs_outside_mask(jax.tree.map(jp.subtract, merged_adam.nu, old_adam.nu), mask),
        ),
    }
    return updates, merged_state, diagnostics
```

Implement `masked_tree_l2_norm` as the square root of the sum of squared finite selected elements and `max_abs_outside_mask` as the maximum absolute unselected value across all leaves; reject an empty PyTree. Implement `frozen_preview_state_drift` with exact tree-structure checks, outside-mask comparisons for actor values and Adam `mu`/`nu`, a complete normalizer mean/variance/count comparison, and a `valid` bit requiring every maximum to equal zero. This helper is used after training to compare each saved checkpoint directly to the migrated E008 parent; per-update zero-drift telemetry proves the same invariant during the compiled loop.

- [ ] **Step 4: Run GREEN and lint**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_preview_adapter.py -q && conda run -n diffsim ruff check src/algorithms/shac/preview_adapter.py tests/test_shac_preview_adapter.py`

Expected: all preview-adapter tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/algorithms/shac/preview_adapter.py tests/test_shac_preview_adapter.py
git commit -m "feat: add exact frozen preview adapter update"
```

### Task 2: Parent Reconstruction and Phase Diagnostics

**Files:**
- Modify: `src/algorithms/shac/preview_adapter.py`
- Modify: `tests/test_shac_preview_adapter.py`

**Interfaces:**
- Consumes: normalized history observations, candidate and zero-preview parent actions, transition phases.
- Produces: `zero_current_preview(observations, *, history_len, legacy_frame_dim, treatment_frame_dim)` and `phase_binned_action_deviation(candidate_actions, parent_actions, phases, *, phase_count, bin_count)`.

- [ ] **Step 1: Write failing reconstruction and bin tests**

```python
def test_zero_current_preview_removes_only_newest_suffix():
    obs = jp.arange(2 * 10 * 328, dtype=jp.float32).reshape(2, -1)
    result = zero_current_preview(obs, history_len=10, legacy_frame_dim=154, treatment_frame_dim=328)
    frames = result.reshape(2, 10, 328)
    np.testing.assert_array_equal(frames[:, :9], obs.reshape(2, 10, 328)[:, :9])
    np.testing.assert_array_equal(frames[:, 9, :154], obs.reshape(2, 10, 328)[:, 9, :154])
    np.testing.assert_array_equal(frames[:, 9, 154:], 0.0)


def test_phase_binned_action_deviation_reports_mean_max_counts_and_validity():
    candidate = jp.array([[[1.0]], [[4.0]], [[8.0]], [[16.0]]])
    parent = jp.zeros_like(candidate)
    result = phase_binned_action_deviation(
        candidate, parent, jp.array([[0], [2], [4], [7]]), phase_count=8, bin_count=2
    )
    np.testing.assert_array_equal(result["bin_counts"], [2, 2])
    np.testing.assert_allclose(result["mean_abs"], [2.5, 12.0])
    np.testing.assert_allclose(result["max_abs"], [4.0, 16.0])
    assert bool(result["valid"])
```

- [ ] **Step 2: Run RED**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_preview_adapter.py -q`

Expected: failures naming both missing functions.

- [ ] **Step 3: Implement reconstruction and finite-aware reductions**

`zero_current_preview` validates the trailing observation dimension equals `history_len * treatment_frame_dim`, reshapes only the trailing dimension, zeros `[..., -1, legacy_frame_dim:]`, and restores the original shape. `phase_binned_action_deviation` computes per-sample mean absolute action difference, maps flattened phases with `phase_bin_indices`, returns fixed-length `bin_counts`, `mean_abs`, `max_abs`, and sets `valid` only when all bins are occupied and all inputs/outputs are finite.

- [ ] **Step 4: Run GREEN and lint**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_preview_adapter.py -q && conda run -n diffsim ruff check src/algorithms/shac/preview_adapter.py tests/test_shac_preview_adapter.py`

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/algorithms/shac/preview_adapter.py tests/test_shac_preview_adapter.py
git commit -m "feat: add preview parent diagnostics"
```

### Task 3: SHAC Trainer Integration and Persistent Checkpoint Metrics

**Files:**
- Modify: `src/algorithms/shac/algorithm.py:40-65,695-1210,1370-1435,1800-2115,2225-2600`
- Modify: `tests/test_shac_exact_resume.py`
- Modify: `tests/test_shac_cagrad_integration.py`

**Interfaces:**
- Consumes: Task-1/2 helper functions.
- Produces: `train(..., actor_preview_adapter: bool = False)`, resume metadata `actor_preview_adapter`, diagnostics in `diag_log.json`, and atomic `checkpoint_phase_metrics.json` rows at midpoint/final checkpoints.

- [ ] **Step 1: Write failing configuration, resume, and persistence tests**

```python
def test_preview_adapter_requires_future_reference_cagrad_and_plain_flax_actor():
    with pytest.raises(ValueError, match="requires future-reference CAGrad"):
        validate_preview_adapter_configuration(
            enabled=True, actor_reference_lookahead_steps=(), actor_cagrad=True,
            history_len=10, source_actor_policy=None, initial_full_actor_policy=None,
        )


def test_preview_adapter_legacy_resume_is_treatment_but_saved_true_is_exact():
    assert resolve_preview_adapter_resume_setting({}, requested=True) is True
    assert resolve_preview_adapter_resume_setting({"actor_preview_adapter": True}, requested=True) is True
    with pytest.raises(ValueError, match="must match"):
        resolve_preview_adapter_resume_setting({"actor_preview_adapter": True}, requested=False)


def test_checkpoint_phase_metrics_are_atomic_and_step_addressed(tmp_path):
    persist_checkpoint_phase_metric(tmp_path, {"step": 1376256, "actor_cagrad_bin_losses": [1, 2, 3, 4, 5]})
    persist_checkpoint_phase_metric(tmp_path, {"step": 1572864, "actor_cagrad_bin_losses": [5, 4, 3, 2, 1]})
    rows = json.loads((tmp_path / "checkpoint_phase_metrics.json").read_text())
    assert [row["step"] for row in rows] == [1376256, 1572864]
```

Add a source-level integration assertion that the adapter branch calls `apply_preview_adapter_update`, assigns `new_actor_norm = state.normalizer`, emits all preview metrics, and records metrics only when `checkpoint_path is not None`.

- [ ] **Step 2: Run RED**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py -q`

Expected: failures naming the missing preview-adapter configuration, resume, persistence, and trainer wiring.

- [ ] **Step 3: Add configuration and exact-resume handling**

Add `actor_preview_adapter: bool = False` to `train`. `resolve_preview_adapter_resume_setting` permits a legacy checkpoint without this key to begin treatment only when explicitly requested, requires a saved boolean otherwise, and rejects changing a saved true treatment to false. `validate_preview_adapter_configuration` requires G1 future reference, CAGrad, ten-frame history, a plain Flax actor, and a true boolean flag. Record `actor_preview_adapter`, `actor_preview_trainable_parameter_count`, `actor_normalizer_frozen`, and `checkpoint_phase_metrics_artifact` in `hparams.json` before any checkpoint.

- [ ] **Step 4: Wire parent-action telemetry into rollout**

When enabled, build the mask once after actor initialization. In `rollout_step`, compute `parent_obs_norm = zero_current_preview(obs_norm, ...)`, apply the same current actor before action noise, and store `candidate_action`, `parent_action`, plus `transition_phase_before_reset(...)`. After population flattening, reduce the two action tensors with `phase_binned_action_deviation`. The disabled path must not add trajectory leaves or alter compilation/output.

- [ ] **Step 5: Wire exact masked optimization and normalizer freezing**

Replace the optimizer call only in the enabled branch:

```python
if actor_preview_adapter:
    updates, new_actor_opt, preview_update_diagnostics = apply_preview_adapter_update(
        actor_opt, grads, state.actor_opt, state.actor_params, preview_adapter_mask
    )
else:
    updates, new_actor_opt = actor_opt.update(grads, state.actor_opt)
new_actor_params = optax.apply_updates(state.actor_params, updates)
```

Set `new_actor_norm = state.normalizer` when enabled; keep the current observation update otherwise. Emit preview gradient/update norms, frozen update and moment drift, normalizer drift, per-bin counts/mean/max action deviation, and a validity bit requiring zeros for every frozen-drift metric and finite occupied bins.

- [ ] **Step 6: Persist checkpoint-aligned objective records**

Implement `persist_checkpoint_phase_metric(save_dir, row)` as an atomic read/replace keyed by integer `step`, sorted by step. Immediately after a periodic checkpoint is returned, persist the current five bin losses, bin counts, action-deviation means/maxima, preview update norm, frozen drift metrics, and validity. Because interval is `196_608`, the 393,216-transition run records exactly midpoint `1_376_256` and final `1_572_864`.

- [ ] **Step 7: Run focused GREEN, legacy neighbor tests, and lint**

Run:

```bash
JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest \
  tests/test_shac_preview_adapter.py \
  tests/test_shac_exact_resume.py \
  tests/test_shac_cagrad.py \
  tests/test_shac_cagrad_integration.py \
  tests/test_shac_future_reference_migration.py \
  tests/test_shac_microbatch.py -q
conda run -n diffsim ruff check src/algorithms/shac/algorithm.py src/algorithms/shac/preview_adapter.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py
```

Expected: all selected tests pass; Ruff reports no errors.

- [ ] **Step 8: Commit**

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py
git commit -m "feat: train frozen G1 preview adapter"
```

### Task 4: Immutable Runner and Objective–Survival Audit

**Files:**
- Create: `tools/run_g1_frozen_preview_adapter_continuation.py`
- Create: `tests/test_g1_frozen_preview_adapter_runner.py`
- Create: `tools/evaluate_preview_objective_survival.py`
- Create: `tests/test_preview_objective_survival_evaluator.py`

**Interfaces:**
- Consumes: `build_cagrad_future_reference_kwargs`, two `checkpoint_phase_metrics.json` rows, and two existing phase-grid `summary.json` files.
- Produces: `build_frozen_preview_adapter_kwargs(...)` and `build_objective_survival_audit(...)` plus CLI JSON output.

- [ ] **Step 1: Write failing immutable-runner tests**

```python
def test_runner_changes_only_preview_adapter_flag_from_future_reference_parent():
    parent = build_cagrad_future_reference_kwargs("fixed_4x5", REF, 0, RESUME)
    candidate = build_frozen_preview_adapter_kwargs("fixed_4x5", REF, 0, RESUME)
    delta = {key: value for key, value in candidate.items() if parent.get(key) != value}
    assert delta == {"actor_preview_adapter": True}
    assert candidate["total_steps"] == 1_572_864
    assert candidate["checkpoint_interval"] == 196_608
```

Verify the parser requires `--solver-profile` and `--resume-from` and exposes no scientific override flags.

- [ ] **Step 2: Write failing pure-audit tests**

```python
def test_audit_pairs_registered_bins_and_reports_rank_agreement():
    records = [
        {"step": 1376256, "actor_cagrad_bin_losses": [5, 4, 3, 2, 1]},
        {"step": 1572864, "actor_cagrad_bin_losses": [4, 3, 2, 1, 0]},
    ]
    summaries = {
        1376256: {"start_phases": [0, 100, 200, 300, 400], "survival_frames": [10, 20, 30, 40, 50]},
        1572864: {"start_phases": [0, 100, 200, 300, 400], "survival_frames": [20, 30, 40, 50, 60]},
    }
    audit = build_objective_survival_audit(records, summaries, phase_count=600)
    assert len(audit["cases"]) == 10
    assert audit["loss_survival_spearman"] < 0.0
    assert audit["valid"]
```

Add rejection tests for missing steps, phase order other than `[0,100,200,300,400]`, nonfinite losses, duplicate checkpoint rows, and a phase-grid checkpoint step that does not match its key.

- [ ] **Step 3: Run RED**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_g1_frozen_preview_adapter_runner.py tests/test_preview_objective_survival_evaluator.py -q`

Expected: collection failures for both missing tools.

- [ ] **Step 4: Implement the runner and pure audit**

The runner imports `build_cagrad_future_reference_kwargs`, copies its dictionary, and sets only `actor_preview_adapter=True`. The audit maps each fixed phase through `phase_bin_indices`, pairs actor loss with `survival_frames`, calculates average ranks without SciPy, and reports `np.corrcoef(loss_ranks, survival_ranks)[0,1]`; lower actor loss should correspond to higher survival, so useful agreement is negative. Include SHA-256 hashes of all three source JSON artifacts in CLI output and write atomically.

- [ ] **Step 5: Run GREEN and lint**

Run:

```bash
JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest \
  tests/test_g1_frozen_preview_adapter_runner.py \
  tests/test_g1_cagrad_future_reference_runner.py \
  tests/test_preview_objective_survival_evaluator.py -q
conda run -n diffsim ruff check \
  tools/run_g1_frozen_preview_adapter_continuation.py \
  tools/evaluate_preview_objective_survival.py \
  tests/test_g1_frozen_preview_adapter_runner.py \
  tests/test_preview_objective_survival_evaluator.py
```

Expected: all selected tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit**

```bash
git add tools/run_g1_frozen_preview_adapter_continuation.py tools/evaluate_preview_objective_survival.py tests/test_g1_frozen_preview_adapter_runner.py tests/test_preview_objective_survival_evaluator.py
git commit -m "feat: add frozen preview experiment tools"
```

### Task 5: Verification and E008 Preflight

**Files:**
- Verify only; repair files are limited to Tasks 1–4 paths.

**Interfaces:**
- Consumes: completed Stage-1 implementation.
- Produces: passing scientific suite and exact pre-update migration/mask evidence.

- [ ] **Step 1: Run the full focused scientific suite once**

Run:

```bash
JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest \
  tests/test_shac_preview_adapter.py \
  tests/test_shac_exact_resume.py \
  tests/test_shac_cagrad.py \
  tests/test_shac_cagrad_integration.py \
  tests/test_shac_future_reference_migration.py \
  tests/test_shac_microbatch.py \
  tests/test_g1_frozen_preview_adapter_runner.py \
  tests/test_g1_cagrad_future_reference_runner.py \
  tests/test_preview_objective_survival_evaluator.py \
  tests/test_g1_tracking_environment.py -q
```

Expected: all selected tests pass. Do not rerun the full suite for a test-only floating tolerance correction; rerun only the failing slice.

- [ ] **Step 2: Run static verification**

Run:

```bash
conda run -n diffsim ruff check src/algorithms/shac/algorithm.py src/algorithms/shac/preview_adapter.py tools/run_g1_frozen_preview_adapter_continuation.py tools/evaluate_preview_objective_survival.py tests/test_shac_preview_adapter.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py tests/test_g1_frozen_preview_adapter_runner.py tests/test_preview_objective_survival_evaluator.py
conda run -n diffsim python -m py_compile src/algorithms/shac/algorithm.py src/algorithms/shac/preview_adapter.py tools/run_g1_frozen_preview_adapter_continuation.py tools/evaluate_preview_objective_survival.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Run exact E008 preflight without training**

Load the registered E008 checkpoint, instantiate the Stage-1 environment/actor, run `migrate_future_reference_train_state`, build the adapter mask, and assert:

```python
assert report["valid"]
assert report["max_action_abs_error"] <= 1e-7
assert trainable_count == 89_088
assert frozen_parameter_max_abs_drift == 0.0
assert frozen_mu_max_abs_drift == 0.0
assert frozen_nu_max_abs_drift == 0.0
assert actor_normalizer_legacy_max_abs_drift == 0.0
```

Save the JSON stdout as the experiment's preflight artifact; do not launch if any assertion fails.

- [ ] **Step 4: Review the branch diff**

Run: `git diff 7571c56..HEAD --stat && git diff 7571c56..HEAD -- src/algorithms/shac/algorithm.py src/algorithms/shac/preview_adapter.py tools/run_g1_frozen_preview_adapter_continuation.py tools/evaluate_preview_objective_survival.py`

Expected: no unrelated environment, solver, reward, reference, or evaluator-policy changes.

### Task 6: Register, Execute, Evaluate, and Curate Stage 1

**Files:**
- Create: `docs/vault/experiments/E-20260811-001-g1-frozen-preview-adapter.md`
- Modify: `experiments/registry.yaml`

**Interfaces:**
- Consumes: exact E008 checkpoint path/hash from registry and Task-5 preflight.
- Produces: registered run directory, midpoint/final phase-grid summaries/videos, `objective_survival_audit.json`, selected checkpoint, and a curated experiment card.

- [ ] **Step 1: Preregister the experiment**

Record hypothesis, exact parent checkpoint/hash, code commit, runner command, two expected checkpoint steps, fixed evaluation phases, Stage-1 gates, invalid-execution conditions, and stop/Stage-2 conditions. Status is `registered`; result fields remain absent until execution.

- [ ] **Step 2: Validate registry and dry-run command**

Run the repository's registry validator command documented in `docs/vault/README.md`, then invoke the runner with `--help` and resolve every input path. Expected: registry valid, exactly one matching E008 parent checkpoint, and no GPU process started by the dry run.

- [ ] **Step 3: Execute Stage 1 alone**

Run `tools/run_g1_frozen_preview_adapter_continuation.py` with `--solver-profile fixed_4x5`, seed 0, the registered reference, exact E008 `--resume-from`, and an experiment-specific output root. Do not run another MJX training process concurrently. Expected checkpoints: `1_376_256` and `1_572_864`.

- [ ] **Step 4: Validate execution artifacts before evaluation**

Assert migration equivalence is valid, exactly two checkpoint-phase metric rows exist at the expected steps, all numeric checkpoint leaves are finite, `actor_preview_adapter=true`, trainable count is `89_088`, all phase bins are occupied, and all CAGrad/preview validity bits are true. Reconstruct the migrated E008 parent once and run `frozen_preview_state_drift` against both saved checkpoints; normalizer/frozen parameter/frozen moment drift must be exactly zero. Any violation marks `invalid-execution` and stops scientific interpretation.

- [ ] **Step 5: Evaluate midpoint and final with the existing phase grid**

Use the same nominal replay-free evaluator and phases `0,100,200,300,400`. Preserve each `summary.json` and video. Select by maximum minimum survival, then median, mean, and earliest checkpoint.

- [ ] **Step 6: Generate objective-survival audit**

Run `tools/evaluate_preview_objective_survival.py` on `checkpoint_phase_metrics.json` and both phase-grid summaries. Expected: ten valid cases, hashes matching immutable inputs, and a finite Spearman coefficient.

- [ ] **Step 7: Apply the preregistered decision**

Advance the selected checkpoint only if minimum survival is at least 21, median at least 48, and all phases meet `[23,50,72,16,40]`. If at least one later phase improves over E008 but a floor regresses, write a separate Stage-2 trust plan with one evidence-pinned scale. If no phase improves, stop the masked-column family and design the separate residual MLP. Do not tune this run.

- [ ] **Step 8: Curate and commit research evidence**

Update the experiment card and registry with actual code/checkpoint/artifact hashes, command, survival vectors, selection, objective-survival audit, validity, and decision. Run registry validation, `git diff --check`, and commit only the card/registry changes with `git commit -m "research: curate G1 frozen preview adapter"`.
