# G1 Zero-Assistance Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue E012 for 64 additional exact-zero-assistance updates and produce eight densely evaluable checkpoints.

**Architecture:** Add one experiment-specific runner that inherits the proven E012 kwargs, changes only the absolute endpoint, validates immutable inputs before compute, and validates exact-zero/frozen-state training artifacts after compute. Reuse the existing no-render phase-grid evaluator and selector after training.

**Tech Stack:** Python 3.11, JAX/MJX, existing SHAC trainer, pytest, Ruff, DiffSim2Real registry.

## Global Constraints

- Resume exact E012 final transition 1720320.
- End at transition 2113536: exactly 64 H12 effective-512 updates.
- Keep assistance start/end at 1327104/1622016, so all new scales are zero.
- Keep checkpoint cadence 49152, yielding eight registered checkpoints.
- Preserve reference reset noise scale 1.0 and all other E012 settings.

---

### Task 1: Exact consolidation runner

**Files:**
- Create: `tools/run_g1_zero_assistance_consolidation.py`
- Create: `tests/test_g1_zero_assistance_consolidation_runner.py`

**Interfaces:**
- Produces: `build_zero_assistance_consolidation_kwargs(...) -> dict`, immutable preflight JSON, training directory, and validation JSON.
- Consumes: E012 final checkpoint, sibling hparams, reference, solver profile, seed, output root, and exact code commit.

- [ ] **Step 1: Write failing tests for exact endpoint/delta, unchanged assistance schedule, expected checkpoint grid, hashes, and exact-zero postvalidation**

- [ ] **Step 2: Run the focused test and confirm RED on the missing module**

Run: `python -m pytest -q tests/test_g1_zero_assistance_consolidation_runner.py`
Expected: collection failure for the missing runner.

- [ ] **Step 3: Implement the minimal inherited runner and fail-closed validations**

```python
kwargs = build_frozen_residual_assistance_kwargs(
    profile_name, reference_path, seed, resume_from
)
kwargs["total_steps"] = 2_113_536
```

- [ ] **Step 4: Run focused and neighboring tests plus Ruff and py_compile**

Run: `python -m pytest -q tests/test_g1_zero_assistance_consolidation_runner.py tests/test_g1_frozen_residual_assistance_runner.py`
Expected: all pass.

- [ ] **Step 5: Commit and request review**

```bash
git add tools/run_g1_zero_assistance_consolidation.py tests/test_g1_zero_assistance_consolidation_runner.py
git commit -m "feat: add zero-assistance consolidation runner"
```

### Task 2: Register, train, evaluate, and curate

**Files:**
- Create: `research/experiments/E-20260812-001.yaml` in the lab registry.
- Modify after result: experiment result, `research/state/current.yaml`, and generated vault projections.

- [ ] **Step 1: Register exact hashes, outcome map, 120-minute training budget, and required artifacts**

- [ ] **Step 2: Validate, dry-run, and execute through `tools.runexp`**

- [ ] **Step 3: Evaluate all eight checkpoints with the existing no-render five-phase evaluator, parallelized over available GPUs**

- [ ] **Step 4: Select by minimum/median/mean/earliest, validate exact-zero evidence, and curate one registered outcome**
