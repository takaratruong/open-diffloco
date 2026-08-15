# Progressive Recovery Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one compact-support ELU residual expert that improves E023's earliest LAFAN failure without changing the frozen parent outside the registered failure corridor.

**Architecture:** Reuse `FrozenPreviewResidualParams` for the immutable E023 parent and zero-head residual adapter. Add a separate frozen `RecoverySupport` artifact and an application wrapper that multiplies the residual by a compact phase-and-state gate. Integrate the wrapper into SHAC and the replay-free evaluators, then execute one registered 32-update experiment with paired parent/candidate selection.

**Tech Stack:** Python 3.11, JAX/JAX NumPy, Flax Linen, Optax, MJX, NumPy, pytest, Ruff, the existing `g1-4x5` solver and plain-YAML research registry.

## Global Constraints

- E023 checkpoint SHA-256 remains `2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f`.
- LAFAN reference SHA-256 remains `bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db`.
- The parent actor, parent optimizer moments, actor normalizer, plant, controller, reward, action transform, and `g1-4x5` solver are immutable.
- The expert is `328 -> 256 -> 29`, ELU, tanh output, zero output-head initialization.
- The support gate is frozen, finite, hash-bound, and exactly zero on every protected negative.
- Training uses H24, effective population 512, 25-percent targeted carried resets, per-environment clip 1.0, Adam `1e-3`, anchor weight 1.0, zero bootstrap, zero assistance, and no CAGrad.
- Do not add observation noise, noisy RSI, pushes, friction randomization, reward changes, hyperparameter sweeps, or post-hoc support-radius tuning.
- Train exactly 32 updates and checkpoint updates 8, 16, and 32.
- Selection requires paired replay-free phase grids, phase-zero improvement of at least three transitions in both repeats, and no loss above two transitions on phases 100/200/300/400.
- Preserve unrelated changes and never edit the Obsidian `.obsidian/` directory.

---

### Task 1: Compact Recovery Support Core

**Files:**
- Create: `src/algorithms/shac/progressive_recovery_expert.py`
- Create: `tests/test_progressive_recovery_expert.py`

**Interfaces:**
- Produces: `RecoverySupport(anchors, radius, phase_min, phase_max, taper)`.
- Produces: `build_recovery_support(positive_frames, negative_frames, positive_phases, taper=4) -> tuple[RecoverySupport, dict[str, object]]`.
- Produces: `compact_recovery_gate(normalized_observations, phases, support) -> jax.Array`.
- Produces: `apply_state_gated_recovery(parent_actor, residual_actor, params, normalized_observations, phases, support, *, history_len, treatment_frame_dim) -> tuple[action, parent_action, gated_residual, gate]`.

- [ ] **Step 1: Write failing construction and gate tests**

```python
def test_support_radius_is_half_nearest_protected_distance():
    positives = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    negatives = np.array([[2.0, 0.0], [3.0, 0.0]])
    support, report = build_recovery_support(
        positives, negatives, np.array([10, 11, 12]), taper=4,
        minimum_positive_coverage=3,
    )
    assert support.radius == pytest.approx(0.9)
    assert report["protected_negative_max_gate"] == 0.0


def test_gate_is_exact_zero_outside_phase_or_state_support():
    gate = compact_recovery_gate(
        jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
        jnp.array([9, 11, 17]),
        support,
    )
    np.testing.assert_array_equal(np.asarray(gate), [0.0, 0.0, 0.0])
```

- [ ] **Step 2: Run the focused RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_progressive_recovery_expert.py`

Expected: collection failure because `progressive_recovery_expert` does not exist.

- [ ] **Step 3: Implement the immutable support and compact kernel**

```python
class RecoverySupport(NamedTuple):
    anchors: jax.Array
    radius: jax.Array
    phase_min: int
    phase_max: int
    taper: int


def compact_recovery_gate(observations, phases, support):
    delta = observations[..., None, :] - support.anchors
    distance = jp.sqrt(jp.min(jp.sum(jp.square(delta), axis=-1), axis=-1))
    state_gate = jp.where(
        distance < support.radius,
        jp.square(1.0 - jp.square(distance / support.radius)),
        0.0,
    )
    left = smoothstep((phases - (support.phase_min - support.taper)) / support.taper)
    right = smoothstep(((support.phase_max + support.taper) - phases) / support.taper)
    return state_gate * left * right
```

`build_recovery_support` must reject nonfinite arrays, wrong ranks, empty sets, radius `<= 0`, or leave-one-out coverage below the requested minimum. It must compute radius as exactly half the minimum protected-negative distance and report all distances and exact-zero checks.

- [ ] **Step 4: Test gated residual application and gradients**

Add tests proving that gate zero is bit-identical to the parent action, gate one matches the existing residual application, support arrays have no gradients or optimizer leaves, and gradients to the expert are finite/nonzero only where the gate is positive.

- [ ] **Step 5: Run focused GREEN and static checks**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_progressive_recovery_expert.py \
  tests/test_shac_residual_preview_adapter.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check \
  src/algorithms/shac/progressive_recovery_expert.py \
  tests/test_progressive_recovery_expert.py
git diff --check
```

Expected: all tests and checks pass.

- [ ] **Step 6: Commit the support core**

```bash
git add src/algorithms/shac/progressive_recovery_expert.py tests/test_progressive_recovery_expert.py
git commit -m "feat: add compact recovery expert support"
```

---

### Task 2: Hash-Bound Support and Targeted-Bank Builder

**Files:**
- Create: `tools/build_g1_progressive_recovery_support.py`
- Create: `tests/test_g1_progressive_recovery_support_builder.py`

**Interfaces:**
- Consumes: E027 bank arrays `actor_obs_history`, `phase`, `source_start_phase`, and all exact state arrays.
- Produces: `build_targeted_bank(source: Mapping[str, np.ndarray], source_phase=0) -> dict[str, np.ndarray]` with exactly 24 rows.
- Produces: `build_support_artifact(positive_frames, negative_frames, positive_phases, ...) -> (npz_payload, json_summary)`.
- Writes atomically: `e023_lafan_phase0_targeted_bank.npz`, `e023_lafan_phase0_recovery_support.npz`, and sibling manifest JSON files.

- [ ] **Step 1: Write failing provenance, slicing, and support tests**

Tests must require exactly 120 source rows, five 24-row source bands, E023/LAFAN/bank hashes, source phase zero, exact preservation of all selected arrays, no pickle-enabled NPZ loading, and manifest-last atomic publication.

- [ ] **Step 2: Run the focused RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_progressive_recovery_support_builder.py`

Expected: import failure for the missing builder.

- [ ] **Step 3: Implement exact protected-state collection**

Load E023 through `tools.evaluate_g1_tracking._load_policy`, construct the pinned LAFAN environment, and use `build_compiled_step`. Roll out exact phases `0/100/200/300/400`. At every pre-reset state, persist phase and the complete ten-frame actor history. Normalize only for the support calculation using E023's frozen normalizer; keep raw histories in the evidence artifact.

Protected negatives are:

```python
phase0 states with phase < p_min
+ states from starts 100/200/300/400 whose phase is in [p_min - 4, p_max + 4]
```

Positive anchors are the 24 E027 bank rows with `source_start_phase == 0`.

- [ ] **Step 4: Implement atomic artifacts and validation**

The support NPZ must contain normalized anchors, radius, phase bounds, taper, leave-one-out distances, protected distances, and SHA-bound source identities. The manifest must independently recompute the exact-zero protected gate and positive-coverage count before publication.

- [ ] **Step 5: Run focused GREEN, Ruff, py_compile, and diff-check**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_progressive_recovery_support_builder.py \
  tests/test_g1_e023_lafan_carried_reset_bank.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check \
  tools/build_g1_progressive_recovery_support.py \
  tests/test_g1_progressive_recovery_support_builder.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
  tools/build_g1_progressive_recovery_support.py
git diff --check
```

- [ ] **Step 6: Commit the builder**

```bash
git add tools/build_g1_progressive_recovery_support.py tests/test_g1_progressive_recovery_support_builder.py
git commit -m "feat: build LAFAN recovery support"
```

---

### Task 3: SHAC Integration and Checkpoint Telemetry

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Create: `tests/test_shac_progressive_recovery_integration.py`
- Modify: `tests/test_shac_residual_preview_adapter.py`

**Interfaces:**
- Consumes: `actor_state_gated_recovery_support_path: str | None` and `allow_resume_actor_state_gated_recovery_start: bool` in `train()`.
- Produces hparams: support path/SHA, radius, phase bounds, taper, protected count, positive coverage, and `actor_state_gated_recovery=True`.
- Produces checkpoint telemetry: gate activation fraction/max, gated residual RMS/max, carried/reference activation fractions, frozen parent drift, and existing action diagnostics.

- [ ] **Step 1: Write failing configuration and integration tests**

Cover fresh-run rejection, missing/changed support on resume, explicit E023 upgrade authority, incompatible use without the residual adapter, invalid support hash, and forwarding of `state.info["phase"]` before reset.

- [ ] **Step 2: Run the focused RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_progressive_recovery_integration.py`

Expected: failures for missing `train` arguments and support loader.

- [ ] **Step 3: Add fail-closed support loading and resume authority**

Add the two `train()` arguments, load the support once before compilation, require `actor_residual_preview_adapter=True`, and persist the exact file SHA. On resume from E023, support and residual-adapter creation require the explicit start flag. On continuation, absent or changed support metadata fails unless a separately named authority is added in a later experiment.

- [ ] **Step 4: Route the gated action through the actor loss**

Replace only the residual call when support is active:

```python
action, parent_action, residual_action, recovery_gate = apply_state_gated_recovery(
    actor,
    residual_preview_actor,
    actor_params,
    obs_norm,
    state.info["phase"],
    recovery_support,
    history_len=actor_history_len,
    treatment_frame_dim=env.actor_frame_obs_dim,
)
```

The gate is computed before `env.step` and before any terminal reset. Preserve the existing ungated path byte-for-byte when the support argument is `None`.

- [ ] **Step 5: Persist complete checkpoint telemetry**

Add transition leaves for `recovery_gate` and `gated_residual_action`. Aggregate finite activation and magnitude statistics across the effective population and save them in every checkpoint row. Validation must reject nonfinite values, activation outside `[0,1]`, activation on protected evidence, zero expert gradient/update norms, or any parent/moment/normalizer drift.

- [ ] **Step 6: Run focused and neighboring GREEN suites**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_shac_progressive_recovery_integration.py \
  tests/test_shac_residual_preview_adapter.py \
  tests/test_g1_e023_lafan_anchored_carried_recovery.py \
  tests/test_shac_microbatch.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check \
  src/algorithms/shac/algorithm.py \
  tests/test_shac_progressive_recovery_integration.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
  src/algorithms/shac/algorithm.py
git diff --check
```

- [ ] **Step 7: Commit the training integration**

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_progressive_recovery_integration.py tests/test_shac_residual_preview_adapter.py
git commit -m "feat: train state-gated recovery experts"
```

---

### Task 4: Replay-Free Paired Evaluation and Evidence

**Files:**
- Modify: `tools/evaluate_g1_flax_phase_grid.py`
- Modify: `tools/evaluate_g1_tracking.py`
- Create: `tools/evaluate_g1_progressive_recovery_pair.py`
- Modify: `tests/test_evaluate_g1_flax_phase_grid.py`
- Create: `tests/test_g1_progressive_recovery_pair.py`

**Interfaces:**
- Phase-grid flag: `--actor-state-gated-recovery-support PATH`.
- Produces per-phase `gate_active_steps`, `gate_activation_fraction`, `gate_max`, `gated_residual_rms`, and support SHA.
- Paired evaluator consumes one parent and three candidate checkpoints and writes two independently executed control/candidate grids plus `selection.json`.

- [ ] **Step 1: Write failing evaluator and selector tests**

Test exact phase forwarding, support/hash mismatch rejection, exact gate-zero parent parity, two-repeat enforcement, the `+3` phase-zero gate, `-2` tolerated floors elsewhere, deterministic ranking, and all four registered outcomes.

- [ ] **Step 2: Run the focused RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_progressive_recovery_pair.py \
  tests/test_evaluate_g1_flax_phase_grid.py
```

- [ ] **Step 3: Extend phase-grid and rendered evaluation**

When the support flag is present, require a `FrozenPreviewResidualParams` checkpoint and its sibling hparams support SHA. Call `apply_state_gated_recovery` with the state's pre-step phase. Record the gate and gated residual before action preparation. The rendered evaluator must place gate activation and support bounds in the summary and NPZ, while retaining the existing MP4/contact-sheet contract.

- [ ] **Step 4: Implement fail-closed paired selection**

For each candidate update `8/16/32`, execute two parent/candidate grids in separate evaluator processes. Eligibility is:

```python
all(candidate_repeat[r][0] >= parent_repeat[r][0] + 3 for r in (0, 1))
and all(
    candidate_repeat[r][phase] >= parent_repeat[r][phase] - 2
    for r in (0, 1) for phase in (1, 2, 3, 4)
)
```

Validate every raw JSON and checkpoint/support SHA in the parent before aggregating. Render only the selected candidate; if none is eligible, render update 32 as diagnostic and label it non-selected.

- [ ] **Step 5: Run focused GREEN and static checks**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_progressive_recovery_pair.py \
  tests/test_evaluate_g1_flax_phase_grid.py \
  tests/test_g1_tracking_evaluator.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check \
  tools/evaluate_g1_flax_phase_grid.py \
  tools/evaluate_g1_tracking.py \
  tools/evaluate_g1_progressive_recovery_pair.py \
  tests/test_g1_progressive_recovery_pair.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
  tools/evaluate_g1_flax_phase_grid.py \
  tools/evaluate_g1_tracking.py \
  tools/evaluate_g1_progressive_recovery_pair.py
git diff --check
```

- [ ] **Step 6: Commit the evaluator**

```bash
git add tools/evaluate_g1_flax_phase_grid.py tools/evaluate_g1_tracking.py \
  tools/evaluate_g1_progressive_recovery_pair.py \
  tests/test_evaluate_g1_flax_phase_grid.py tests/test_g1_progressive_recovery_pair.py
git commit -m "feat: evaluate progressive recovery experts"
```

---

### Task 5: Guarded E029 Runner, Registry, Execution, and Curation

**Files:**
- Create: `tools/run_g1_progressive_recovery_expert.py`
- Create: `tests/test_g1_progressive_recovery_runner.py`
- Create in lab repository: `research/experiments/E-20260815-029.yaml`
- Modify after validated evidence: `research/state/current.yaml`
- Modify only if promotion criteria are met: `docs/vault/findings/F-20260810-001 Phase Scalarization Redistributes Early G1 Competence.md`

**Interfaces:**
- Runner builds the exact treatment kwargs and validates three archives at steps `1671168`, `1769472`, and `1966080`.
- Runner emits `preflight.json`, support/bank artifacts, `training_validation.json`, paired phase grids, `selection.json`, rendered rollout/NPZ/summary/contact sheet, and action/gate learning curves.

- [ ] **Step 1: Write failing runner contract tests**

Assert the only scientific changes from E027 are targeted support/bank, gated expert application, no CAGrad, and the 32-update budget. Require exact E023/reference/model/controller/code/support hashes, physical-256/two-shard effective-512 H24, carried probability 0.25, anchor 1.0, actor LR `1e-3`, clip 1.0, zero bootstrap/assistance/noisy RSI/domain randomization, and checkpoints at the three exact steps.

- [ ] **Step 2: Run the focused RED, implement the runner, and run GREEN**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_progressive_recovery_runner.py
```

Implement preflight, builder invocation, guarded training, complete telemetry validation, paired evaluation, selection, and manifest-last publication. Rerun until the focused test passes.

- [ ] **Step 3: Run the bounded verification suite**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_progressive_recovery_expert.py \
  tests/test_g1_progressive_recovery_support_builder.py \
  tests/test_shac_progressive_recovery_integration.py \
  tests/test_g1_progressive_recovery_pair.py \
  tests/test_g1_progressive_recovery_runner.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check \
  src/algorithms/shac/progressive_recovery_expert.py \
  src/algorithms/shac/algorithm.py \
  tools/build_g1_progressive_recovery_support.py \
  tools/evaluate_g1_progressive_recovery_pair.py \
  tools/run_g1_progressive_recovery_expert.py
git diff --check
```

- [ ] **Step 4: Commit code and register E029**

Commit the runner/tests in the code repository. In the lab repository, create E029 with the exact new code commit, dirty-patch hash, support outcome map, command, one GPU, seed zero, 32-update budget, and required artifacts. Run:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260815-029 --dry-run
```

Commit only the E029 registry record.

- [ ] **Step 5: Execute one guarded experiment**

Run: `uv run python -m tools.runexp E-20260815-029`

Do not launch another experiment while E029 runs. Monitor the first compiled update for finite support activation, nonzero expert gradients, zero frozen drift, and checkpoint publication. Stop on any invalidity.

- [ ] **Step 6: Evaluate and curate**

Independently verify every registered artifact and preregistered outcome. Update E029 and `research/state/current.yaml` from the same evidence, promote the existing finding only if the result establishes a reusable boundary, run registry validation, rebuild the vault, and publish the selected or diagnostic MP4/PNG to the Dashboard preview path.

- [ ] **Step 7: Final verification**

Run:

```bash
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py build-vault
test -s docs/vault/attachments/research-previews/E-20260815-029/evaluation.mp4
test -s docs/vault/attachments/research-previews/E-20260815-029/contact_sheet.png
```

Report the exact paired vectors, gate activation, expert action magnitude, selected checkpoint or failure outcome, what remains unproven, and the next registered decision boundary.
