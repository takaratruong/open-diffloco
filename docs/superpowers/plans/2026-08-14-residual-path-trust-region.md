# Residual-Path Behavioral Trust-Region Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic diagnostic checkpoints on two residual-parameter paths and evaluate whether either contains a componentwise non-regressive long-dance policy.

**Architecture:** A small builder owns fail-closed parameter projection/interpolation and atomic pickle/JSON publication. The existing compiled/clipped phase-grid evaluator remains unchanged and consumes the generated checkpoints.

**Tech Stack:** Python 3.11, JAX/Flax pytrees, NumPy, pickle, pytest, Ruff.

## Global Constraints

- Do not mutate source or target checkpoints.
- Preserve the source parent, normalizer, critic, environment, and optimizer state exactly.
- Remove only the target adapter's assistance row, and require zero-scale action parity before interpolation.
- Alphas are exactly `0.125, 0.25, 0.5, 0.75`.
- No GPU training or physics change is authorized by this plan.

---

### Task 1: Pure residual interpolation and artifact builder

**Files:**
- Create: `tools/build_g1_residual_path_checkpoints.py`
- Create: `tests/test_build_g1_residual_path_checkpoints.py`

**Interfaces:**
- Consumes: two SHAC `TrainState` pickle files and an ordered alpha tuple.
- Produces: `project_zero_scale_target_adapter(source_params, target_params)`, `interpolate_residual_actor_params(source_params, target_params, alpha)`, and atomic diagnostic checkpoint/manifest artifacts.

- [ ] **Step 1: Write failing tests** for exact frozen-tree validation, 329-to-328 row removal, alpha endpoints/interior arithmetic, nonfinite rejection, zero-scale action parity, deterministic output naming, and atomic manifest-last publication.
- [ ] **Step 2: Run RED** with `/home/ubuntu/miniconda3/envs/diffsim/bin/pytest -q tests/test_build_g1_residual_path_checkpoints.py`; expect import failure for the missing builder.
- [ ] **Step 3: Implement the minimal builder** using existing `FrozenPreviewResidualParams`, `PreviewResidualAdapter`, and `apply_frozen_preview_residual`; use `Path.replace` for manifest-last publication and `pickle.HIGHEST_PROTOCOL` for checkpoints.
- [ ] **Step 4: Run GREEN** with the same focused command; expect all tests to pass.
- [ ] **Step 5: Run neighboring checks**: `pytest -q tests/test_shac_residual_preview_adapter.py tests/test_evaluate_g1_flax_phase_grid.py`, Ruff on the two new files, `py_compile`, and `git diff --check`.
- [ ] **Step 6: Commit** only the builder, tests, design, and plan.

### Task 2: Registered interpolation discriminator

**Files:**
- Create in lab repo: `research/experiments/E-20260814-015.yaml`
- Create at runtime: `runs/E-20260814-015/path_checkpoints/`, `phase_grid/`, and `selection.json`

**Interfaces:**
- Consumes: retained E013 step `1867776`, E014 aware step `2359296`, and E014 blind step `2457600`.
- Produces: eight immutable interpolated checkpoints, eight corrected five-phase summaries, and one preregistered outcome.

- [ ] **Step 1: Register** the exact source/target hashes, code commit, alpha grid, componentwise E013 floor, output map, GPUs, and wall-time budget.
- [ ] **Step 2: Validate and dry-run** with `researchctl.py validate` and `tools.runexp ... --dry-run`.
- [ ] **Step 3: Build checkpoints** with the committed builder and verify every output hash against its manifest.
- [ ] **Step 4: Evaluate in parallel** with the existing `tools/evaluate_g1_flax_phase_grid.py` and corrected clipped/compiled boundary.
- [ ] **Step 5: Select** only a componentwise non-regressive candidate; otherwise record no safe path.
- [ ] **Step 6: Render and curate** the selected candidate only if one is eligible; validate registry, rebuild vault, and update the phase-redistribution finding.
