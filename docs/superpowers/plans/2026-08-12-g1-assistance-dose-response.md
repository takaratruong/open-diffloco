# G1 Assistance Dose-Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one bounded evaluation-only matrix that determines whether E012 reduced the torso assistance required for suffix completion.

**Architecture:** A pure analysis module validates the registered matrix and classifies threshold trends. A single-checkpoint evaluator reuses the validated torso-wrench and frozen residual-policy seams. A JAX-free parent runner assigns four physical GPUs, launches one worker per checkpoint, fails fast, and atomically aggregates their artifacts.

**Tech Stack:** Python 3.11, JAX/MJX, MuJoCo, NumPy, pytest, Ruff, existing DiffSim2Real registry/runexp.

## Global Constraints

- Evaluation only: no optimizer updates, policy selection, rendering, or training-path changes.
- Checkpoints are fixed at transitions 1327104, 1474560, 1622016, and 1720320.
- Phases are exactly `(0, 100, 200, 300, 400)` and scales exactly `(0.0, 0.1, 0.25, 0.5, 1.0)`.
- Each checkpoint's 25 conditions execute in one process on one physical GPU.
- Output is finite, provenance-bound, atomic, and fail-closed.

---

### Task 1: Pure dose-response contract

**Files:**
- Create: `src/evaluation/g1_assistance_dose_response.py`
- Test: `tests/test_g1_assistance_dose_response.py`

**Interfaces:**
- Produces: `required_scale(records, *, scales) -> float | None` and `classify_threshold_trajectory(checkpoints) -> str`.
- Consumes records containing `scale`, `completed_reference_suffix`, and finite telemetry validity.

- [ ] **Step 1: Write failing tests for minimum scale and all three scientific classifications**

```python
def test_required_scale_returns_smallest_completion():
    records = [
        {"scale": 0.0, "completed_reference_suffix": False, "valid": True},
        {"scale": 0.1, "completed_reference_suffix": True, "valid": True},
        {"scale": 0.25, "completed_reference_suffix": True, "valid": True},
    ]
    assert required_scale(records, scales=(0.0, 0.1, 0.25)) == 0.1
```

- [ ] **Step 2: Run the test and confirm RED because the module is absent**

Run: `python -m pytest -q tests/test_g1_assistance_dose_response.py`
Expected: collection failure naming `g1_assistance_dose_response`.

- [ ] **Step 3: Implement strict validation, threshold extraction, and trajectory classification**

```python
def required_scale(records, *, scales):
    by_scale = {float(record["scale"]): record for record in records}
    if set(by_scale) != set(scales) or not all(r["valid"] for r in records):
        raise ValueError("dose-response records do not satisfy the grid")
    return next((scale for scale in scales
                 if by_scale[scale]["completed_reference_suffix"]), None)
```

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `python -m pytest -q tests/test_g1_assistance_dose_response.py`
Expected: all tests pass.

- [ ] **Step 5: Commit the pure contract**

```bash
git add src/evaluation/g1_assistance_dose_response.py tests/test_g1_assistance_dose_response.py
git commit -m "feat: add assistance dose-response contract"
```

### Task 2: Single-checkpoint evaluator

**Files:**
- Create: `tools/evaluate_g1_assistance_dose_response.py`
- Modify: `tests/test_g1_assistance_dose_response.py`

**Interfaces:**
- Consumes: one checkpoint, reference, output path, code commit, checkpoint label, and frozen scales/phases.
- Produces: one atomic JSON with 25 rollout records and five required-scale values.

- [ ] **Step 1: Add failing tests for immutable parser defaults, exact grid coverage, bit-exact zero wrench, and manifest validation**

```python
def test_grid_has_every_phase_scale_pair():
    assert registered_conditions() == tuple(
        (phase, scale) for phase in PHASES for scale in SCALES
    )
```

- [ ] **Step 2: Run focused tests and confirm RED on the missing evaluator API**

Run: `python -m pytest -q tests/test_g1_assistance_dose_response.py`
Expected: import failure or missing-symbol assertion.

- [ ] **Step 3: Implement the evaluator by reusing `frozen_e008_environment_kwargs`, `load_frozen_e008_policy`, `rollout_condition`, and wrench telemetry**

```python
for phase, scale in registered_conditions():
    initial, _ = paired_reset(env, phase=phase, seed=args.seed)
    summary, trace = rollout_condition(
        env, initial_state=initial, action_fn=action_fn, phase=phase,
        torso_body_id=torso_body_id, torso_slot=torso_slot,
        parameters=parameters, scale=scale, profile=profile,
    )
```

- [ ] **Step 4: Run focused tests, Ruff, py_compile, and a no-GPU provenance dry probe**

Run: `python -m pytest -q tests/test_g1_assistance_dose_response.py tests/test_g1_frozen_torso_wrench_oracle.py tests/test_g1_torso_wrench_oracle.py`
Expected: all tests pass.

Run: `ruff check src/evaluation/g1_assistance_dose_response.py tools/evaluate_g1_assistance_dose_response.py tests/test_g1_assistance_dose_response.py`
Expected: success.

- [ ] **Step 5: Commit the evaluator**

```bash
git add tools/evaluate_g1_assistance_dose_response.py tests/test_g1_assistance_dose_response.py
git commit -m "feat: evaluate fixed assistance dose response"
```

### Task 3: Four-GPU bounded runner

**Files:**
- Create: `tools/run_g1_assistance_dose_response.py`
- Create: `tests/test_g1_assistance_dose_response_runner.py`

**Interfaces:**
- Consumes: four `label=checkpoint` arguments, four GPU IDs, reference, code commit, and output root.
- Produces: `preflight.json`, four worker JSON files, logs, and `assistance_dose_response.json`.

- [ ] **Step 1: Write failing tests for exact hashes, device cardinality, hard-crash fail-fast behavior, artifact validation, and aggregate classification**

```python
def test_runner_requires_one_distinct_gpu_per_checkpoint():
    with pytest.raises(ValueError, match="distinct physical GPU"):
        validate_devices((1, 1, 3, 5), checkpoint_count=4)
```

- [ ] **Step 2: Run the runner test and confirm RED because the module is absent**

Run: `python -m pytest -q tests/test_g1_assistance_dose_response_runner.py`
Expected: collection failure naming the missing runner.

- [ ] **Step 3: Implement JAX-free preflight, subprocess launch, peer termination, artifact validation, and atomic aggregation**

```python
processes[label] = subprocess.Popen(
    evaluator_command(spec), env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
)
```

- [ ] **Step 4: Run focused tests and static verification**

Run: `python -m pytest -q tests/test_g1_assistance_dose_response_runner.py tests/test_g1_assistance_dose_response.py`
Expected: all tests pass.

Run: `ruff check tools/run_g1_assistance_dose_response.py tests/test_g1_assistance_dose_response_runner.py`
Expected: success.

- [ ] **Step 5: Commit the runner**

```bash
git add tools/run_g1_assistance_dose_response.py tests/test_g1_assistance_dose_response_runner.py
git commit -m "feat: run assistance dose response in parallel"
```

### Task 4: Review, register, execute, and curate

**Files:**
- Create: `research/experiments/E-20260812-000.yaml` in the lab registry.
- Modify after evaluation: `research/state/current.yaml`, the experiment result, and generated vault projections.

**Interfaces:**
- Consumes: reviewed clean code commit and the four exact checkpoint hashes.
- Produces: one registered scientific outcome and its next decision.

- [ ] **Step 1: Run the complete focused suite and request code review**

Run: `JAX_ENABLE_X64=1 python -m pytest -q tests/test_g1_assistance_dose_response.py tests/test_g1_assistance_dose_response_runner.py tests/test_g1_frozen_torso_wrench_oracle.py tests/test_g1_torso_wrench_oracle.py`
Expected: all tests pass.

- [ ] **Step 2: Register the experiment with exact outcome map, hashes, devices, four checkpoints, 150-minute wall time, and zero training transitions**

- [ ] **Step 3: Validate and dry-run the registry contract**

Run: `uv run python tools/researchctl.py validate`
Expected: `registry valid`.

Run: `uv run python -m tools.runexp E-20260812-000 --dry-run`
Expected: approved experiment command and run directory without compute.

- [ ] **Step 4: Execute the guarded experiment and monitor all four workers**

Run: `uv run python -m tools.runexp E-20260812-000`
Expected: one aggregate artifact or a fail-closed invalid execution.

- [ ] **Step 5: Curate exactly one preregistered result and rebuild the vault**

Run: `uv run python tools/researchctl.py validate && uv run python tools/researchctl.py build-vault`
Expected: valid registry and updated generated projections while user Obsidian files remain unstaged.

