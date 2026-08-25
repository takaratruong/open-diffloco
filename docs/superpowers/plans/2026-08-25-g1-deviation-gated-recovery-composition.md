# G1 Deviation-Gated Recovery Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate whether a fixed continuous body-error gate can compose E026's learned recovery residual with E023 without regressing ordinary short-walk behavior.

**Architecture:** Add one pure deviation-gate/composition module beside the existing compact-support recovery module. Build one experiment-specific evaluator that loads the exact E026 composite checkpoint, executes parent/global/gated arms from identical resets, validates and atomically publishes raw evidence, classifies the result, and renders only the validated gated phase-zero arm.

**Tech Stack:** Python 3.11, JAX, Flax, MJX/MuJoCo, NumPy, pytest, Ruff, existing G1 evaluator and registry tooling.

## Global Constraints

- Gate is exactly zero at error `<= 0.10`, smoothstep on `(0.10, 0.20)`, and exactly one at error `>= 0.20`.
- No checkpoint parameter, normalizer, plant, controller, reference, solver, action boundary, seed, gradient, or optimizer change.
- Evaluate phases `0/25/50/75/100` in one process from matched exact reset/RNG state.
- Persist raw parent, residual, gate, composed, sampled, and effective actions plus state and tracking telemetry.
- Publish evidence atomically and manifest-last; render only after numerical validation.
- Do not treat finite clip completion as stable walking unless the final-ten-frame trend gate passes.
- Do not sweep thresholds or train in this experiment.

---

### Task 1: Pure deviation gate and action composition

**Files:**
- Create: `src/algorithms/shac/deviation_gated_recovery.py`
- Create: `tests/test_shac_deviation_gated_recovery.py`

**Interfaces:**
- Consumes: JAX arrays for previous-transition body-position error, parent action, and residual action.
- Produces: `DeviationGate(lower: float, upper: float)`, `deviation_recovery_gate(error, contract) -> jax.Array`, and `compose_deviation_gated_recovery(parent_action, residual_action, error, contract) -> tuple[action, gated_residual, gate]`.

- [ ] **Step 1: Write failing endpoint, interior, shape, and nonfinite tests**

```python
def test_deviation_gate_has_exact_registered_endpoints():
    errors = jnp.asarray([0.0, 0.10, 0.15, 0.20, 1.0])
    gate = deviation_recovery_gate(errors, REGISTERED_DEVIATION_GATE)
    np.testing.assert_array_equal(np.asarray(gate[[0, 1]]), [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(gate[2]), 0.5, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(gate[[3, 4]]), [1.0, 1.0])

def test_deviation_gate_rejects_nonfinite_error():
    with pytest.raises(ValueError, match="finite"):
        deviation_recovery_gate(jnp.asarray(np.nan), REGISTERED_DEVIATION_GATE)

def test_composition_preserves_parent_at_zero_and_global_at_one():
    parent = jnp.asarray([[1.0, 2.0]])
    residual = jnp.asarray([[0.5, -0.25]])
    zero, zero_residual, zero_gate = compose_deviation_gated_recovery(
        parent, residual, jnp.asarray([0.0]), REGISTERED_DEVIATION_GATE
    )
    full, full_residual, full_gate = compose_deviation_gated_recovery(
        parent, residual, jnp.asarray([0.2]), REGISTERED_DEVIATION_GATE
    )
    np.testing.assert_array_equal(zero, parent)
    np.testing.assert_array_equal(zero_residual, np.zeros_like(parent))
    np.testing.assert_array_equal(full, parent + residual)
    np.testing.assert_array_equal(full_residual, residual)
```

- [ ] **Step 2: Run RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_deviation_gated_recovery.py`

Expected: collection fails because `src.algorithms.shac.deviation_gated_recovery` does not exist.

- [ ] **Step 3: Implement the minimal pure module**

```python
class DeviationGate(NamedTuple):
    lower: float
    upper: float

REGISTERED_DEVIATION_GATE = DeviationGate(lower=0.10, upper=0.20)

def deviation_recovery_gate(error, contract=REGISTERED_DEVIATION_GATE):
    values = jp.asarray(error)
    if not bool(np.asarray(jp.all(jp.isfinite(values)))):
        raise ValueError("body-position error must be finite")
    ratio = jp.clip((values - contract.lower) / (contract.upper - contract.lower), 0.0, 1.0)
    smooth = ratio * ratio * (3.0 - 2.0 * ratio)
    return jp.where(values <= contract.lower, 0.0, jp.where(values >= contract.upper, 1.0, smooth))

def compose_deviation_gated_recovery(parent_action, residual_action, error, contract=REGISTERED_DEVIATION_GATE):
    parent = jp.asarray(parent_action)
    residual = jp.asarray(residual_action)
    if parent.shape != residual.shape:
        raise ValueError("parent and residual action shapes must match")
    gate = deviation_recovery_gate(error, contract)
    gated = gate[..., None] * residual
    action = jp.where(gate[..., None] == 0.0, parent, parent + gated)
    return action, gated, gate
```

- [ ] **Step 4: Run GREEN and static checks**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_deviation_gated_recovery.py`

Expected: all tests pass.

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check src/algorithms/shac/deviation_gated_recovery.py tests/test_shac_deviation_gated_recovery.py && python -m py_compile src/algorithms/shac/deviation_gated_recovery.py`

- [ ] **Step 5: Commit**

```bash
git add src/algorithms/shac/deviation_gated_recovery.py tests/test_shac_deviation_gated_recovery.py
git commit -m "feat: add deviation-gated recovery composition"
```

### Task 2: Frozen three-arm evaluator and evidence validator

**Files:**
- Create: `tools/evaluate_g1_deviation_gated_recovery.py`
- Create: `tests/test_g1_deviation_gated_recovery.py`
- Reuse: `tools/evaluate_g1_tracking.py`
- Reuse: `src/algorithms/shac/residual_preview_adapter.py`

**Interfaces:**
- Consumes: exact E026 composite checkpoint/hparams, pinned short-walk reference, model/controller assets, code commit, output directory, and seed zero.
- Produces: `preflight.json`, fifteen arm/phase NPZ+JSON records, `selection.json`, `gated_phase0.mp4`, `gated_phase0_contact_sheet.png`, and manifest-last `completion.json`.

- [ ] **Step 1: Write failing evaluator-contract tests**

```python
def test_classification_requires_componentwise_preservation():
    result = classify_deviation_gate(
        parent=[116, 99, 67, 49, 24],
        global_arm=[124, 99, 74, 49, 24],
        gated=[124, 98, 74, 49, 24],
        gated_metrics=nominal_metrics(),
        final_tail=stable_tail(),
    )
    assert result == "useful-correction-not-localizable"

def test_short_clip_solution_requires_stable_tail():
    result = classify_deviation_gate(
        parent=[116, 99, 67, 49, 24],
        global_arm=[124, 99, 74, 49, 24],
        gated=[124, 99, 74, 49, 24],
        gated_metrics=nominal_metrics(),
        final_tail=np.linspace(0.05, 0.25, 10),
    )
    assert result == "deviation-gating-advances"

def test_manifest_rejects_tampered_raw_evidence(tmp_path):
    manifest = publish_fixture(tmp_path)
    raw = next(tmp_path.glob("arms/*/*.npz"))
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        validate_completion_manifest(manifest)
```

- [ ] **Step 2: Run RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_deviation_gated_recovery.py`

Expected: collection fails because the evaluator module does not exist.

- [ ] **Step 3: Implement evaluator seams before the real rollout**

Implement these exact pure/public seams in `tools/evaluate_g1_deviation_gated_recovery.py`:

```python
PHASES = (0, 25, 50, 75, 100)
ARMS = ("parent", "global", "gated")

def classify_deviation_gate(*, parent, global_arm, gated, gated_metrics, final_tail) -> str:
    parent = np.asarray(parent, dtype=np.int64)
    global_arm = np.asarray(global_arm, dtype=np.int64)
    gated = np.asarray(gated, dtype=np.int64)
    preserved = bool(np.all(gated >= parent))
    improved = bool(np.any(gated > parent))
    stable_tail = bool(np.all(np.diff(np.asarray(final_tail, dtype=np.float64)) <= 0.0))
    metrics_valid = bool(gated_metrics["within_five_percent"])
    if preserved and improved and metrics_valid and stable_tail and np.all(gated == np.asarray([124, 99, 74, 49, 24])):
        return "deviation-gating-solves-short-clip"
    if preserved and improved and metrics_valid:
        return "deviation-gating-advances"
    if np.any(global_arm > parent):
        return "useful-correction-not-localizable"
    return "correction-intrinsically-insufficient"

def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)

def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".npz", delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
```

`validate_raw_rollout` must open without pickle, require the exact registered
array-key set, check equal row counts and finite numeric arrays, recompute
`gate`, `gated_residual`, and `candidate_action`, and reject nonmonotone phase or
done/terminal disagreement. `validate_completion_manifest` must require exactly
the fifteen `(arm, phase)` entries, recompute every file SHA-256, call
`validate_raw_rollout` for each entry, recompute classification from raw
summaries, and reject unexpected files or provenance drift.

- [ ] **Step 4: Implement the matched real rollout**

Load E026 once through the existing checkpoint loader. Extract the frozen parent and residual adapter. Reset once per phase, clone the exact reset state into all three arms, and execute each arm with no noise/randomization/replay/assistance. Compute the action immediately before `env.step`:

```python
_, parent_action, residual_action = apply_frozen_preview_residual(
    parent_actor,
    residual_actor,
    actor_params,
    normalized_observation,
    history_len=env.actor_history_len,
    treatment_frame_dim=env.actor_frame_obs_dim,
)
if arm == "parent":
    candidate = parent_action
elif arm == "global":
    candidate = parent_action + residual_action
else:
    candidate, gated_residual, gate = compose_deviation_gated_recovery(
        parent_action,
        residual_action,
        state.metrics["body_position_error"],
    )
effective = prepare_evaluation_action(env, candidate)
next_state = compiled_step(state, effective)
```

Persist pre-step error, gate, all action layers, state, and post-step metrics so the validator can independently reproduce the boundary.

- [ ] **Step 5: Add rendering only after validated numerical selection**

Reuse the existing MuJoCo paired renderer and contact-sheet helper. Select proof
indices with `np.linspace(0, len(qpos) - 1, 8, dtype=np.int64)`, verify every
proof frame is nonempty and full-body, then encode the complete gated phase-zero
MP4.

- [ ] **Step 6: Run GREEN, neighboring tests, and CPU smoke**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_deviation_gated_recovery.py tests/test_shac_deviation_gated_recovery.py tests/test_g1_tracking_evaluator.py`

Expected: all tests pass.

Run one no-render, max-one-step CPU smoke against the exact E026 checkpoint and require `publication_complete=false`; this smoke must not be interpreted scientifically.

- [ ] **Step 7: Commit**

```bash
git add tools/evaluate_g1_deviation_gated_recovery.py tests/test_g1_deviation_gated_recovery.py
git commit -m "feat: evaluate deviation-gated G1 recovery"
```

### Task 3: Review, register, and execute one experiment

**Files:**
- Create in lab repository: `research/experiments/E-20260825-000.yaml` or the next free same-date ID.
- Modify after evaluation only: `research/state/current.yaml`

**Interfaces:**
- Consumes: clean reviewed code commit and the exact E026 checkpoint/hparams/assets.
- Produces: one approved experiment record and one immutable guarded run under `runs/<ID>/`.

- [ ] **Step 1: Run focused code review and repair only Critical/Important findings**

Review provenance, matched reset/RNG, pre/post transition alignment, parent/global/gated identity, action boundary, classification precedence, atomic publication, and render causality. Re-run only affected focused tests after repairs.

- [ ] **Step 2: Create one experiment YAML with exact outcome map**

Use `research_thread: g1-tracking-shac`, controlled tags for closed-loop transfer and phase competence, parent `E-20260815-026`, one GPU, seed zero, and a 30-minute wall-time budget. Pin clean code commit, empty dirty-patch SHA, checkpoint/hparams/reference/model/controller hashes, conda environment, and g1-4x5 solver.

- [ ] **Step 3: Validate and dry-run**

Run: `uv run python tools/researchctl.py validate`

Run: `uv run python -m tools.runexp <EXPERIMENT_ID> --dry-run`

Expected: both exit zero and the dry run resolves exactly one seed and one GPU.

- [ ] **Step 4: Approve and launch**

Set `approved_by: codex` and an exact UTC timestamp, then run:

`uv run python -m tools.runexp <EXPERIMENT_ID>`

Do not launch any second experiment while this run is active.

### Task 4: Curate evidence and publish the visible result

**Files:**
- Modify: `research/experiments/<EXPERIMENT_ID>.yaml`
- Modify: `research/state/current.yaml`
- Generated: `docs/vault/Dashboard.md`
- Generated/uncommitted: `docs/vault/attachments/research-previews/<EXPERIMENT_ID>/`

**Interfaces:**
- Consumes: validated `completion.json`, raw rollouts, selection, MP4, and contact sheet.
- Produces: one canonical verdict, current decision boundary, rebuilt newest-first dashboard, and visible local media.

- [ ] **Step 1: Reopen and validate every required artifact**

Run the evaluator's manifest validator in a fresh process and independently compare the three survival vectors and selected outcome with the raw NPZ records.

- [ ] **Step 2: Inspect the contact sheet before publishing the video**

Confirm policy/reference identity, full-body visibility, correct time order, no hidden reset/replay/assistance, and agreement between visible terminal posture and summary metrics.

- [ ] **Step 3: Curate the exact preregistered outcome**

Update only the experiment result and current decision implied by the classifier. Do not call a clip completion stable walking unless the final-tail gate passes.

- [ ] **Step 4: Rebuild the vault and publish media**

Run: `uv run python tools/researchctl.py build-vault`

Copy canonical MP4/PNG evidence into `docs/vault/attachments/research-previews/<EXPERIMENT_ID>/`, preserve canonical copies under `runs/`, and verify the generated Dashboard embeds resolve with the newest experiment first.

- [ ] **Step 5: Validate and commit only owned lab records**

Run: `uv run python tools/researchctl.py validate`

Run focused registry/dashboard tests, `git diff --check`, then commit the experiment, current state, qualifying finding if warranted, and generated projection files without touching `.obsidian` or unrelated user changes.
