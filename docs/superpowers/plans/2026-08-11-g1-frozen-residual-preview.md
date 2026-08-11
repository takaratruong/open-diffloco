# G1 Frozen Residual-Preview Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a zero-effect nonlinear preview-to-action residual whose parameters are the only trainable actor state, then run a dense replay-free capacity experiment from exact E008.

**Architecture:** Wrap the append-only migrated Flax parent and a 328-to-256-to-29 ELU residual adapter in one PyTree. Reuse the existing exact masked-Adam, frozen-normalizer, five-bin CAGrad, checkpoint telemetry, and dense phase-grid paths; add only composite-policy migration/application, trainer/evaluator wiring, and an immutable runner.

**Tech Stack:** Python 3.11, JAX, Flax Linen, Optax, NumPy, pytest, Ruff, pickle/JSON artifacts.

## Global Constraints

- Resume exact E008 at transition `1_179_648` and use delta previews at offsets `(4, 8, 12)`.
- Use history length `10`, treatment frame width `328`, adapter hidden width `256`, action width `29`, and exactly `91_677` trainable adapter scalars.
- Initialize the adapter output kernel and bias to exact zeros; pre-update candidate actions must match the migrated parent within `1e-7`.
- Freeze every parent actor value, parent Adam `mu`/`nu` value, and actor-normalizer value. Only adapter parameters and their moments may change.
- Preserve H12, 256 physical environments, accumulation factor two, effective batch 512, five-bin CAGrad, 4x5 solver, zero termination margin, seed 0, and 393,216 additional transitions.
- Keep the feature default-off. Do not unfreeze the parent, sweep adapter width, change the learning rate, add a teacher, or add a second scientific treatment.

---

## File Map

- Create `src/algorithms/shac/residual_preview_adapter.py`: composite policy type, residual module, pure application, optimizer migration, mask, and migration audit.
- Create `tests/test_shac_residual_preview_adapter.py`: numerical contracts for composition, migration, and masking.
- Modify `src/algorithms/shac/algorithm.py`: explicit configuration/resume validation, composite migration, rollout, adapter-only update, telemetry, and hparams.
- Modify `tests/test_shac_exact_resume.py`: legacy-start and treated-resume contracts.
- Modify `tests/test_shac_cagrad_integration.py`: default-off and trainer-wiring contracts.
- Modify `tools/evaluate_g1_flax_phase_grid.py`: replay-free composite-policy evaluation and metadata.
- Modify `tests/test_evaluate_g1_flax_phase_grid.py`: evaluator composition and payload tests.
- Create `tools/run_g1_frozen_residual_preview_continuation.py`: immutable E008 treatment runner.
- Create `tests/test_g1_frozen_residual_preview_runner.py`: exact runner delta and parser tests.

### Task 1: Pure composite policy and migration

**Files:**
- Create: `src/algorithms/shac/residual_preview_adapter.py`
- Create: `tests/test_shac_residual_preview_adapter.py`

**Interfaces:**
- Produces `FrozenPreviewResidualParams(parent, adapter)`, `PreviewResidualAdapter(action_dim=29, hidden_dim=256)`, `apply_frozen_preview_residual(parent_actor, residual_actor, params, normalized_observations, *, history_len, treatment_frame_dim)`, `build_residual_adapter_mask(params)`, `migrate_frozen_preview_residual_state(state, *, actor_optimizer, parent_actor, residual_actor, adapter_key, history_len, treatment_frame_dim)`, and `residual_adapter_migration_report(...)`.

- [ ] **Step 1: Write failing composition and slicing tests**

```python
def test_zero_head_preserves_parent_and_uses_only_current_frame():
    adapter = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
    params = adapter.init(jax.random.PRNGKey(3), jp.zeros((1, 5)))
    history = jp.arange(15.0).reshape(1, 15)
    residual = adapter.apply(params, history.reshape(1, 3, 5)[:, -1])
    np.testing.assert_array_equal(residual, jp.zeros((1, 2)))


def test_composite_action_has_feedback_gradient_but_no_parent_parameter_gradient():
    action = apply_frozen_preview_residual(
        parent_actor, residual_actor, params, observations,
        history_len=3, treatment_frame_dim=5,
    )
    np.testing.assert_allclose(action, parent_actor.apply(params.parent, observations))
    assert float(jp.linalg.norm(jax.grad(loss)(params).adapter["params"]["Dense_1"]["bias"])) > 0.0
    assert all(float(jp.linalg.norm(x)) == 0.0 for x in jax.tree.leaves(jax.grad(loss)(params).parent))
    assert float(jp.linalg.norm(jax.grad(lambda obs: jp.sum(apply_policy(params, obs)))(observations))) > 0.0
```

- [ ] **Step 2: Run RED**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_residual_preview_adapter.py -q`

Expected: collection fails because `src.algorithms.shac.residual_preview_adapter` does not exist.

- [ ] **Step 3: Implement the minimal composite policy**

```python
class FrozenPreviewResidualParams(NamedTuple):
    parent: Any
    adapter: Any


class PreviewResidualAdapter(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, frame):
        hidden = nn.elu(nn.Dense(self.hidden_dim)(frame))
        logits = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(hidden)
        return jp.tanh(logits)
```

`apply_frozen_preview_residual` validates the trailing width, stop-gradients only the parent parameter tree, evaluates the parent on the full normalized history, evaluates the adapter on `reshape(..., history_len, treatment_frame_dim)[..., -1, :]`, and returns `(candidate_action, parent_action, residual_action)`.

- [ ] **Step 4: Write failing optimizer-migration and mask tests**

The tests initialize a real chained clip-plus-Adam state with nonzero parent moments, migrate it, and assert: the composite mask has exactly `91_677` true entries, every parent parameter and moment is byte-exact, every adapter moment is zero, the Adam count and schedule state are inherited, the initial residual is zero, and mutation of any parent leaf invalidates the migration report.

- [ ] **Step 5: Implement migration and audit, then run GREEN**

Initialize the composite optimizer template, replace its parent `mu`/`nu` with the inherited parent trees, keep zero adapter moments, and copy the Adam count plus schedule state. Build the mask as false over `parent` and true over `adapter`. Return a state replacement without changing any non-actor leaf.

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_residual_preview_adapter.py -q && conda run -n diffsim ruff check src/algorithms/shac/residual_preview_adapter.py tests/test_shac_residual_preview_adapter.py`

Expected: all focused tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit**

```bash
git add src/algorithms/shac/residual_preview_adapter.py tests/test_shac_residual_preview_adapter.py
git commit -m "feat: add frozen residual preview policy"
```

### Task 2: SHAC integration and exact resume boundary

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tests/test_shac_exact_resume.py`
- Modify: `tests/test_shac_cagrad_integration.py`

**Interfaces:**
- Produces `train(..., actor_residual_preview_adapter: bool = False, actor_residual_preview_hidden: int = 256)`, hparams keys of the same names, `residual_adapter_migration.json`, and existing checkpoint telemetry with adapter-only counts and exact frozen drift.

- [ ] **Step 1: Write failing configuration and resume tests**

Tests require the new treatment to be default-off, mutually exclusive with `actor_preview_adapter`, restricted to future-reference CAGrad with ten-frame Flax actors, able to start only from an explicitly authorized no-preview future-reference upgrade, and required to match the saved flag/width on treated resume.

- [ ] **Step 2: Run RED**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py -q`

Expected: failures name the absent flags, resolver, and trainer wiring.

- [ ] **Step 3: Wire the treatment without changing the disabled path**

After the existing future-reference migration and report, call `migrate_frozen_preview_residual_state`, persist its report, and set the existing masked-update boundary to the adapter-only mask. In `actor_loss`, use `apply_frozen_preview_residual` only under the new flag; reuse candidate/parent phase diagnostics. Freeze the actor normalizer and reuse `apply_preview_adapter_update`, CAGrad, finite telemetry, checkpoint metrics, and frozen-drift gates whenever either frozen treatment is enabled. Persist `actor_residual_preview_trainable_parameter_count=91677`.

- [ ] **Step 4: Run GREEN and neighboring tests**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py tests/test_shac_preview_adapter.py tests/test_shac_future_reference_migration.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py
git commit -m "feat: integrate residual preview training"
```

### Task 3: Immutable runner and replay-free evaluator

**Files:**
- Create: `tools/run_g1_frozen_residual_preview_continuation.py`
- Create: `tests/test_g1_frozen_residual_preview_runner.py`
- Modify: `tools/evaluate_g1_flax_phase_grid.py`
- Modify: `tests/test_evaluate_g1_flax_phase_grid.py`

**Interfaces:**
- The runner changes E007's dense frozen contract only by replacing the linear preview adapter with `actor_residual_preview_adapter=True`, `actor_residual_preview_hidden=256`, and keeping delta preview mode.
- The evaluator accepts `--actor-residual-preview-adapter`, applies the same composite helper, and records adapter kind, width, parameter count, checkpoint hash, reference hash, solver, and preview mode.

- [ ] **Step 1: Write failing exact-delta runner tests**

```python
parent = build_frozen_delta_preview_kwargs("g1-4x5", reference, 0, checkpoint)
candidate = build_frozen_residual_preview_kwargs("g1-4x5", reference, 0, checkpoint)
assert candidate["actor_residual_preview_adapter"] is True
assert candidate["actor_residual_preview_hidden"] == 256
assert candidate["actor_preview_adapter"] is False
assert candidate["actor_reference_preview_mode"] == "delta"
assert {k: v for k, v in candidate.items() if k not in changed} == {
    k: v for k, v in parent.items() if k not in changed
}
```

- [ ] **Step 2: Write failing evaluator parity test**

Create a tiny composite checkpoint, compute one action with the training helper, run the evaluator action builder on the same normalized observation, and require exact equality. Require the payload to record `actor_residual_preview_adapter: true`, hidden width `256`, and trainable count `91677`.

- [ ] **Step 3: Run RED**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_g1_frozen_residual_preview_runner.py tests/test_evaluate_g1_flax_phase_grid.py -q`

Expected: failures name the missing runner and evaluator option.

- [ ] **Step 4: Implement runner and evaluator support, then run GREEN**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_g1_frozen_residual_preview_runner.py tests/test_evaluate_g1_flax_phase_grid.py -q && conda run -n diffsim ruff check tools/run_g1_frozen_residual_preview_continuation.py tools/evaluate_g1_flax_phase_grid.py tests/test_g1_frozen_residual_preview_runner.py tests/test_evaluate_g1_flax_phase_grid.py`

Expected: all focused tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add tools/run_g1_frozen_residual_preview_continuation.py tools/evaluate_g1_flax_phase_grid.py tests/test_g1_frozen_residual_preview_runner.py tests/test_evaluate_g1_flax_phase_grid.py
git commit -m "feat: run and evaluate residual preview treatment"
```

### Task 4: Integrated verification and experiment pin

**Files:**
- Review all Task 1-3 files.
- Create the canonical `E-20260811-008` record only after the code commit is clean.

- [ ] **Step 1: Run the focused integrated suite**

Run: `JAX_ENABLE_X64=1 conda run -n diffsim python -m pytest tests/test_shac_residual_preview_adapter.py tests/test_shac_preview_adapter.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py tests/test_shac_future_reference_migration.py tests/test_g1_frozen_residual_preview_runner.py tests/test_evaluate_g1_flax_phase_grid.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run static verification**

Run: `conda run -n diffsim ruff check src/algorithms/shac/residual_preview_adapter.py src/algorithms/shac/algorithm.py tools/run_g1_frozen_residual_preview_continuation.py tools/evaluate_g1_flax_phase_grid.py tests/test_shac_residual_preview_adapter.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py tests/test_g1_frozen_residual_preview_runner.py tests/test_evaluate_g1_flax_phase_grid.py && python -m py_compile src/algorithms/shac/residual_preview_adapter.py src/algorithms/shac/algorithm.py tools/run_g1_frozen_residual_preview_continuation.py tools/evaluate_g1_flax_phase_grid.py && git diff --check`

Expected: Ruff, compilation, and diff checks succeed.

- [ ] **Step 3: Review the exact commit and pin the experiment**

Inspect every changed hunk for disabled-path drift, silent resume upgrades, evaluator/trainer mismatch, unmasked parent moments, and incomplete provenance. Register one 120-minute, one-GPU E008 treatment with eight dense checkpoints, replay-free phases 0/100/200/300/400, floors 23/50/72/16/40, and advancement key strictly greater than 27/54/53.0.
