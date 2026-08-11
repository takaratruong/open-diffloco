# G1 Frozen Residual Muon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one optimizer-only Muon treatment for the E008 frozen nonlinear residual and expose an immutable runner for a causal replay-free comparison.

**Architecture:** Keep the composite actor and policy forward path unchanged. Split only the adapter optimizer: apply the existing norm-1 global clip once, send the 328-by-256 input kernel through official Optax Muon, send the three remaining adapter arrays through unchanged Optax Adam, and retain the parent Adam state as an immutable snapshot.

**Tech Stack:** Python 3.11, JAX, Flax Linen, Optax 0.2.8, NumPy, pytest, Ruff, pickle/JSON artifacts.

## Global Constraints

- Start from the exact E008 no-preview checkpoint at transition `1_179_648`; do not continue the trained E-20260811-008 adapter.
- Preserve delta previews `(4, 8, 12)`, history `10`, frame width `328`, hidden width `256`, action width `29`, ELU, and the exact-zero output head.
- Apply one global norm-`1.0` clip before optimizer partitioning.
- Apply Muon only to `params/Dense_0/kernel` with beta `0.95`, five Newton-Schulz steps, Nesterov momentum, Frobenius preconditioning, zero weight decay, and `consistent_rms=0.2`.
- Apply unchanged Adam (`b1=0.9`, `b2=0.999`) to `Dense_0/bias`, `Dense_1/kernel`, and `Dense_1/bias` using the same actor schedule.
- Inherit the parent optimizer and schedule counts, initialize every adapter momentum to zero, and preserve the parent optimizer snapshot byte-exact.
- Preserve H12, 256 physical environments, accumulation two, effective batch 512, five-bin CAGrad, fixed 4x5 solver, seed 0, zero termination margin, domain randomization, action noise, 393,216 transitions, and eight checkpoints.
- Keep `actor_residual_preview_optimizer="adam"` as the default. Do not sweep any Muon or architecture hyperparameter.

---

## File Map

- Modify `src/algorithms/shac/residual_preview_adapter.py`: adapter partition types, official Muon/Adam optimizer construction, count migration, adapter-only update, and migration audit.
- Modify `tests/test_shac_residual_preview_adapter.py`: discriminating optimizer partition, clipping, count, zero-moment, finite-update, and frozen-snapshot tests.
- Modify `src/algorithms/shac/algorithm.py`: optimizer option validation/resume handling, Muon initialization/update branch, telemetry, and hparams.
- Modify `tests/test_shac_exact_resume.py`: default, invalid value, legacy start, and treated-resume metadata contracts.
- Modify `tests/test_shac_cagrad_integration.py`: static trainer wiring and metadata contracts.
- Create `tools/run_g1_frozen_residual_muon_continuation.py`: immutable optimizer-only treatment runner.
- Create `tests/test_g1_frozen_residual_muon_runner.py`: exact runner delta and parser tests.

### Task 1: Pure residual Muon optimizer boundary

**Files:**
- Modify: `src/algorithms/shac/residual_preview_adapter.py`
- Modify: `tests/test_shac_residual_preview_adapter.py`

**Interfaces:**
- Consumes: `FrozenPreviewResidualParams`, an inherited clip-plus-Adam parent state, the adapter parameter tree, and an actor scalar schedule.
- Produces: `ResidualAdapterAuxParams`, `FrozenPreviewResidualMuonState`, `build_residual_muon_optimizers(schedule)`, `initialize_residual_muon_optimizer(...)`, `apply_residual_muon_update(...)`, and `residual_muon_migration_report(...)`.

- [ ] **Step 1: Write failing parameter-partition tests**

```python
def test_residual_muon_partition_round_trips_adapter_tree():
    adapter = PreviewResidualAdapter(action_dim=3, hidden_dim=5).init(
        jax.random.PRNGKey(7), jp.zeros((1, 11), dtype=jp.float32)
    )
    kernel, auxiliary = split_residual_adapter_params(adapter)
    assert kernel.shape == (11, 5)
    assert auxiliary.dense0_bias.shape == (5,)
    assert auxiliary.dense1_kernel.shape == (5, 3)
    assert auxiliary.dense1_bias.shape == (3,)
    rebuilt = merge_residual_adapter_params(adapter, kernel, auxiliary)
    assert_tree_exact(rebuilt, adapter)
```

Also require rejection of missing or wrong-rank `Dense_0/kernel` and preservation
of `dict` versus `FrozenDict` container type.

- [ ] **Step 2: Run RED**

Run: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_residual_preview_adapter.py -q`

Expected: import failures name the absent partition types/functions.

- [ ] **Step 3: Implement the minimal split/merge types**

```python
class ResidualAdapterAuxParams(NamedTuple):
    dense0_bias: jax.Array
    dense1_kernel: jax.Array
    dense1_bias: jax.Array


class FrozenPreviewResidualMuonState(NamedTuple):
    parent_optimizer_state: optax.OptState
    muon_state: optax.OptState
    adam_state: optax.OptState
```

`split_residual_adapter_params` extracts only the four registered arrays.
`merge_residual_adapter_params` copies the template through Flax
`freeze`/`unfreeze`, replaces those arrays, and restores the original container
type. Do not accept extra trainable adapter leaves silently.

- [ ] **Step 4: Write failing optimizer initialization tests**

Create a real inherited clip-plus-Adam parent state whose count and schedule
count are nonzero. Require:

```python
muon_opt, adam_opt = build_residual_muon_optimizers(schedule)
state = initialize_residual_muon_optimizer(
    muon_optimizer=muon_opt,
    adam_optimizer=adam_opt,
    parent_optimizer_state=parent_state,
    adapter_params=composite.adapter,
)
assert_tree_exact(state.parent_optimizer_state, parent_state)
assert optimizer_counts(state.muon_state) == [parent_count, parent_count]
assert optimizer_counts(state.adam_state) == [parent_count, parent_count]
assert optimizer_momenta_are_zero(state.muon_state)
assert optimizer_momenta_are_zero(state.adam_state)
```

Inspect the initialized state to require that only the input kernel is present
in the Muon branch and the output kernel remains in the Adam auxiliary tree.

- [ ] **Step 5: Implement official Muon plus unchanged Adam construction**

```python
def build_residual_muon_optimizers(schedule):
    muon = optax.contrib.muon(
        learning_rate=schedule,
        ns_steps=5,
        beta=0.95,
        weight_decay=0.0,
        nesterov=True,
        adaptive=False,
        preconditioning="frobenius",
        consistent_rms=0.2,
    )
    adam = optax.adam(schedule, b1=0.9, b2=0.999)
    return muon, adam
```

Initialize Muon on the extracted two-dimensional kernel and Adam on
`ResidualAdapterAuxParams`. Replace only `MuonState`, `ScaleByAdamState`, and
`ScaleByScheduleState` counts with the inherited parent Adam count; leave
Newton-Schulz coefficients and all zero moment arrays unchanged.

- [ ] **Step 6: Write failing update and migration-audit tests**

Use gradients with global norm greater than one. Compare the auxiliary update
against a direct `optax.adam` call on the once-globally-clipped auxiliary
gradient. Require the Muon update to be finite and different from Adam on the
same matrix, parent updates to be exact zeros, the parent optimizer snapshot to
remain exact, and diagnostics to report finite nonzero Muon/auxiliary update
norms. Mutating the parent snapshot or resetting any inherited count must make
the migration report invalid.

- [ ] **Step 7: Implement the adapter-only update and report, then run GREEN**

`apply_residual_muon_update` must call
`optax.clip_by_global_norm(1.0).update` exactly once on `gradients.adapter`,
split the clipped tree, update the two optimizers, merge adapter updates, and
return `FrozenPreviewResidualParams(parent=zeros, adapter=updates)` plus the
new wrapped state. It must preserve the parent snapshot verbatim and expose the
existing four preview diagnostic keys plus Muon-kernel and auxiliary-Adam
gradient/update norms.

Run: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_residual_preview_adapter.py -q && /home/ubuntu/miniconda3/envs/diffsim/bin/ruff check src/algorithms/shac/residual_preview_adapter.py tests/test_shac_residual_preview_adapter.py`

Expected: all focused tests pass and Ruff reports no errors.

- [ ] **Step 8: Commit**

```bash
git add src/algorithms/shac/residual_preview_adapter.py tests/test_shac_residual_preview_adapter.py
git commit -m "feat: add residual Muon optimizer boundary"
```

### Task 2: SHAC configuration, resume, and update integration

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tests/test_shac_exact_resume.py`
- Modify: `tests/test_shac_cagrad_integration.py`

**Interfaces:**
- Consumes: Task 1's optimizer builders, initializer, update helper, state type, and migration report.
- Produces: `train(..., actor_residual_preview_optimizer: str = "adam")`, strict resume metadata, `residual_muon_migration.json`, checkpoint telemetry, and fixed Muon hparams.

- [ ] **Step 1: Write failing option and resume tests**

Require the train signature default to `"adam"`; allow only `"adam"` and
`"muon"`; reject Muon without the residual adapter; allow a Muon legacy start
only through the existing explicit future-reference upgrade; and require an
exact saved optimizer match for treated resumes. A saved residual checkpoint
without the new key resolves to `"adam"` for backward compatibility.

- [ ] **Step 2: Run RED**

Run: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py -q`

Expected: failures name the absent optimizer argument, resolver, validation,
and trainer wiring.

- [ ] **Step 3: Add strict pure configuration helpers**

Add the optimizer argument next to the existing residual flags. Extend
`validate_residual_preview_adapter_configuration` with the string value and
reject Muon unless the residual treatment is enabled. Extend
`resolve_residual_preview_adapter_resume_setting` to return
`(enabled, hidden_dim, optimizer_name)` and enforce saved metadata exactly,
while mapping a missing historical key to `adam`.

- [ ] **Step 4: Write failing initialization/update/metadata integration tests**

Require source inspection and pure helper tests to prove:

```python
if actor_residual_preview_optimizer == "muon":
    muon_actor_opt, residual_adam_opt = build_residual_muon_optimizers(actor_schedule)
    composite_optimizer_state = initialize_residual_muon_optimizer(...)
else:
    composite_optimizer_state = initialize_residual_adapter_optimizer(...)
```

and the train step dispatches `apply_residual_muon_update` only for the Muon
residual. Require hparams to persist optimizer name, beta `0.95`, NS steps `5`,
Nesterov `true`, preconditioning `frobenius`, and consistent RMS `0.2`.

- [ ] **Step 5: Implement the Muon branch without changing Adam/default paths**

Build the two adapter optimizers only after the actor schedule exists. During
future-upgrade residual migration, initialize the wrapped Muon state and emit
`residual_muon_migration.json`; retain the existing Adam migration/report for
the default branch. During actor update, dispatch the Task 1 helper only when
the saved optimizer name is Muon. Reuse all CAGrad, normalizer, checkpoint,
finiteness, parent-drift, and evaluator behavior unchanged.

- [ ] **Step 6: Run GREEN and neighboring tests**

Run: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py tests/test_shac_residual_preview_adapter.py tests/test_shac_preview_adapter.py tests/test_shac_future_reference_migration.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py
git commit -m "feat: integrate residual Muon training"
```

### Task 3: Immutable Muon runner and integrated verification

**Files:**
- Create: `tools/run_g1_frozen_residual_muon_continuation.py`
- Create: `tests/test_g1_frozen_residual_muon_runner.py`
- Review: all Task 1-2 files and existing residual evaluator.

**Interfaces:**
- Consumes: `build_frozen_residual_preview_kwargs(...)` and Task 2's train option.
- Produces: `build_frozen_residual_muon_kwargs(...)` and a CLI identical to E008 except for optimizer assignment.

- [ ] **Step 1: Write the failing exact-delta runner test**

```python
parent = build_frozen_residual_preview_kwargs("g1-4x5", reference, 0, checkpoint)
candidate = build_frozen_residual_muon_kwargs("g1-4x5", reference, 0, checkpoint)
assert candidate["actor_residual_preview_optimizer"] == "muon"
changed = {"actor_residual_preview_optimizer"}
assert {k: v for k, v in candidate.items() if k not in changed} == {
    k: v for k, v in parent.items() if k not in changed
}
```

Also require `--resume-from`, seed, reference, output root, and solver parsing to
match the E008 runner.

- [ ] **Step 2: Run RED**

Run: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_g1_frozen_residual_muon_runner.py -q`

Expected: collection fails because the runner module does not exist.

- [ ] **Step 3: Implement the minimal immutable runner**

Subclass by composition: call `build_frozen_residual_preview_kwargs`, update
only `actor_residual_preview_optimizer="muon"`, and preserve the existing
`configure_jax`, solver context, output-root, and returned-save-directory path.

- [ ] **Step 4: Run GREEN and the complete focused suite**

Run: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_residual_preview_adapter.py tests/test_shac_preview_adapter.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py tests/test_shac_future_reference_migration.py tests/test_g1_frozen_residual_preview_runner.py tests/test_g1_frozen_residual_muon_runner.py tests/test_evaluate_g1_flax_phase_grid.py -q`

Expected: all focused and neighboring tests pass.

- [ ] **Step 5: Run static verification**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check src/algorithms/shac/residual_preview_adapter.py src/algorithms/shac/algorithm.py tools/run_g1_frozen_residual_preview_continuation.py tools/run_g1_frozen_residual_muon_continuation.py tests/test_shac_residual_preview_adapter.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py tests/test_g1_frozen_residual_preview_runner.py tests/test_g1_frozen_residual_muon_runner.py && /home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile src/algorithms/shac/residual_preview_adapter.py src/algorithms/shac/algorithm.py tools/run_g1_frozen_residual_muon_continuation.py && git diff --check`

Expected: Ruff, Python compilation, and diff checks succeed.

- [ ] **Step 6: Self-review and commit**

Inspect every changed hunk for double clipping, accidental Muon use on the
output head, reset schedule counts, parent-moment decay, default-path drift,
silent resume upgrades, incomplete metadata, and runner deltas beyond the one
optimizer key.

```bash
git add tools/run_g1_frozen_residual_muon_continuation.py tests/test_g1_frozen_residual_muon_runner.py
git commit -m "feat: add residual Muon experiment runner"
```

### Task 4: Canonical experiment registration and launch gate

**Files:**
- Create in canonical repo: `research/experiments/E-20260811-009.yaml`
- Update in canonical repo: `research/state/current.yaml`
- Generate in canonical repo: `docs/vault/experiments/E-20260811-009.md`, `docs/vault/02 Current State.md`, and `docs/vault/hypotheses/H-SCI-001.md`.

**Interfaces:**
- Consumes: the clean code commit from Task 3 and E008's exact parent/reference/protocol.
- Produces: one approved, dry-run-valid, one-GPU E009 contract ready for guarded execution.

- [ ] **Step 1: Pin the clean implementation commit and hashes**

Record `git rev-parse HEAD`, require an empty tracked diff, compute the empty
patch SHA-256, and recompute the exact E008 parent and reference hashes. The
command must call `tools/run_g1_frozen_residual_muon_continuation.py` and write
under `$DIFFSIM_SEED_DIR`.

- [ ] **Step 2: Write the E009 causal contract**

Register one change: residual optimizer `adam` to `muon`. Require 64 H12
updates, effective population 512, eight checkpoints, finite Muon and auxiliary
Adam telemetry, exact frozen state, and replay-free vectors at phases
0/100/200/300/400. Advancement requires floors 70/63/95/70/44 and a key
strictly greater than 44/70/68.4; finite no-gain retains E008 Adam; invalid
execution authorizes only repair.

- [ ] **Step 3: Validate, dry-run, build the vault, and run focused registry tests**

Run: `uv run python tools/researchctl.py validate && uv run python -m tools.runexp E-20260811-009 --dry-run && uv run python tools/researchctl.py build-vault && uv run pytest tests/test_registry.py -q -k 'not typed_registry_wikilink_targets_exist'`

Expected: registry valid, dry-run prints the exact pinned command, the vault
build succeeds, and seven focused registry tests pass with the known historical
typed-wikilink test deselected.

- [ ] **Step 4: Commit and launch exactly once**

Commit only E009/current-state/generated projection files, then run:

```bash
uv run python -m tools.runexp E-20260811-009
```

Do not launch another scientific experiment until E009 training, audit, dense
replay-free evaluation, and curation are complete.
