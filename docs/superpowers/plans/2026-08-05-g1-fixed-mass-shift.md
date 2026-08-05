# G1 Fixed-Mass Differentiable Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one deterministic G1 body-mass/inertia shift seam and use it to run a preregistered, visible differentiable residual-adaptation experiment.

**Architecture:** Reuse SHAC's existing `mass_range` transport but restrict the G1 environment to an equal fixed pair. Apply the scalar once to the MuJoCo plant before MJX conversion, pass it through the existing training and evaluation tools, and preserve it in evidence summaries.

**Tech Stack:** Python 3.10, MuJoCo/MJX 3.9, JAX x64, Flax SHAC, `unittest`, `researchctl`.

## Global Constraints

- The only scientific plant change is fixed non-world `body_mass` and `body_inertia` scale `1.15`.
- Nominal scale defaults to `1.0`.
- G1 rejects randomized or invalid `mass_range` values.
- Training uses seed `0`, 256 environments, horizon `12`, 196,608 steps, fixed checkpoint step `64,512`, and the exact E-20260805-074 optimizer/controller contract.
- No Reliable Claude, mass sweep, checkpoint search, or collocation work is in scope.
- No scientific GPU execution occurs before registry approval.

---

### Task 1: Fixed G1 Plant-Mass Boundary

**Files:**
- Modify: `tests/test_g1_tracking_environment.py`
- Modify: `src/envs/g1_tracking/environment.py`

**Interfaces:**
- Consumes: existing SHAC `mass_range: tuple`
- Produces: `G1TrackingEnv.body_mass_scale: float` and a scaled `mj_model`/`mjx_model`

- [ ] **Step 1: Write failing model-scaling tests**

Add tests that construct nominal and `mass_range=(1.15, 1.15)` environments,
compare their pre-MJX MuJoCo arrays, assert world entries are unchanged, and
assert every non-world `body_mass` and `body_inertia` value has ratio `1.15`.
Add table-driven invalid cases for unequal, nonpositive, nonfinite, and
wrong-length ranges.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_tracking_environment.G1TrackingEnvironmentTest.test_fixed_mass_scale_changes_only_non_world_mass_and_inertia \
  tests.test_g1_tracking_environment.G1TrackingEnvironmentTest.test_mass_scale_rejects_invalid_or_randomized_ranges
```

Expected: failure because G1 currently ignores `mass_range`.

- [ ] **Step 3: Implement the fixed scale**

Add the explicit constructor parameter, validate an equal finite positive pair,
store `body_mass_scale`, and multiply `mj_model.body_mass[1:]` plus
`mj_model.body_inertia[1:]` before calling `mjx.put_model`.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run the command from Step 2. Expected: both tests pass.

- [ ] **Step 5: Commit the isolated plant seam**

```bash
git add src/envs/g1_tracking/environment.py tests/test_g1_tracking_environment.py
git commit -m "Add fixed G1 body mass shift"
```

### Task 2: Training And Evaluation Transport

**Files:**
- Modify: `tests/test_g1_tracking_runner.py`
- Modify: `tests/test_g1_tracking_evaluator.py`
- Modify: `tools/run_g1_tracking_rmr50_shac.py`
- Modify: `tools/evaluate_g1_tracking.py`
- Modify: `tools/compare_g1_tracking_residual.py`

**Interfaces:**
- Consumes: `body_mass_scale: float`
- Produces: fixed `mass_range=(scale, scale)` in training and the same model in paired/rendered evaluation

- [ ] **Step 1: Write failing transport tests**

Assert that `build_train_kwargs(..., body_mass_scale=1.15)` returns
`mass_range == (1.15, 1.15)`. Assert that
`make_evaluation_env(..., body_mass_scale=1.15)` exposes
`env.body_mass_scale == 1.15`. Preserve existing default-scale assertions.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_tracking_runner \
  tests.test_g1_tracking_evaluator
```

Expected: new keyword arguments are not accepted.

- [ ] **Step 3: Implement CLI and function transport**

Add a positive `--body-mass-scale` argument to the training, paired comparison,
and render CLIs. Thread it through `build_train_kwargs`,
`make_evaluation_env`, and the comparator environment construction. Add
`body_mass_scale` to render summaries and paired comparison documents.

- [ ] **Step 4: Run focused and adjacent tests**

```bash
JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_tracking_runner \
  tests.test_g1_tracking_evaluator \
  tests.test_g1_tracking_residual_comparison \
  tests.test_shac_checkpoint_archival
```

Expected: all tests pass.

- [ ] **Step 5: Run the complete G1 environment suite**

```bash
JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_tracking_environment
```

Expected: all tests pass.

- [ ] **Step 6: Commit the transport**

```bash
git add tools/run_g1_tracking_rmr50_shac.py tools/evaluate_g1_tracking.py \
  tools/compare_g1_tracking_residual.py tests/test_g1_tracking_runner.py \
  tests/test_g1_tracking_evaluator.py
git commit -m "Thread fixed mass shift through G1 tools"
```

### Task 3: Preregister And Execute Shifted Training

**Files:**
- Create: `research/experiments/E-20260805-083.yaml` in the active `diffsim2real-lab` worktree
- Modify: `research/state/current.yaml` in the active `diffsim2real-lab` worktree

**Interfaces:**
- Consumes: pinned implementation commit and the design's scientific contract
- Produces: durable step-addressed shifted residual checkpoints and training diagnostics

- [ ] **Step 1: Verify and push the implementation commit**

Run focused tests from Tasks 1 and 2, `git diff --check`, inspect the code diff,
and push the code branch. Record the exact commit SHA.

- [ ] **Step 2: Register the experiment before GPU work**

Create E-20260805-083 with the exact code SHA, model/reference hashes,
`body_mass_scale=1.15`, full training command, environment, seed, walltime,
fixed checkpoint selection, required artifacts, and preregistered outcome map.
Run:

```bash
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py dry-run E-20260805-083
```

Expected: registry valid and dry-run succeeds.

- [ ] **Step 3: Execute once through the guarded runner**

```bash
uv run python tools/researchctl.py runexp E-20260805-083
```

Expected: return code zero, no timeout, durable step-64,512 checkpoint, finite
diagnostics, and no unregistered rerun.

- [ ] **Step 4: Independently verify artifacts**

Hash `diag_log.json`, `hparams.json`, and the fixed checkpoint. Count numeric
checkpoint leaves and independently require all finite. Treat absent or
inconsistent artifacts as invalid execution.

### Task 4: Paired Shift Evaluation And Visible Evidence

**Files:**
- Create: `research/experiments/E-20260805-084.yaml` in the active `diffsim2real-lab` worktree
- Create: `research/experiments/E-20260805-085.yaml` in the active `diffsim2real-lab` worktree
- Modify: `research/hypotheses/H-G1-024.yaml`
- Modify: `research/state/current.yaml`
- Modify: `docs/vault/findings/F-20260805-001 Residual SHAC Preserves the G1 Walking Basin but Nominal Gains Are Seed-Sensitive.md`

**Interfaces:**
- Consumes: exact E-20260805-083 step-64,512 checkpoint
- Produces: shifted same-process source/residual comparison, video, montage, and a scientific verdict

- [ ] **Step 1: Preregister the paired evaluation**

Pin phases 0/30/60/90, seed 0, mass scale 1.15, solver 4/5, 60-step suffixes,
source checkpoint, residual checkpoint SHA, and all improvement thresholds.
Validate and dry-run before execution.

- [ ] **Step 2: Run the paired comparison once**

Execute through `researchctl runexp`. Independently report source and residual
terminal counts, aggregate reward delta, all six error deltas, and all four
phase reward deltas.

- [ ] **Step 3: Preregister and render the exact checkpoint**

Render 60 phase-zero steps through the shifted environment, create a
first/middle/final montage, inspect it visually, and verify trajectory/video
agreement.

- [ ] **Step 4: Curate and publish**

Set experiment verdicts from the preregistered outcome maps, update the
hypothesis and current state, rebuild only the relevant Obsidian projection
files without staging unrelated dirty files, validate the registry, commit, and
push both repositories.

- [ ] **Step 5: Decide the next mechanism**

If shifted adaptation passes, retain residual SHAC as the first working
differentiable sim-to-sim adaptation method and design the collocation-transfer
test. If it preserves but misses improvement, localize the exact shifted
objective/gradient. If it degrades or the source falls, preregister one
mechanistically justified correction rather than sweeping mass or checkpoints.
