# G1 Fixed-Mass Headroom Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare nominal source, shifted source, and shifted residual in one process and quantify the learned residual action.

**Architecture:** Extend the existing paired evaluator only when an optional nominal baseline scale is supplied. Reuse its rollout and aggregation boundaries, load the source actor once, and publish a backward-compatible enriched JSON document.

**Tech Stack:** Python 3.10, JAX x64, MuJoCo/MJX 3.9, NumPy, RSL-RL checkpoint loader, `unittest`, `researchctl`.

## Global Constraints

- No Reliable Claude is used.
- Policy, checkpoint, shift magnitude, phases, seed, solver, reward, and termination remain fixed.
- The material threshold remains exactly `0.001`.
- No scientific execution occurs before registration.

---

### Task 1: Baseline Triad And Residual-Action Summary

**Files:**
- Modify: `tests/test_g1_tracking_residual_comparison.py`
- Modify: `tools/compare_g1_tracking_residual.py`

**Interfaces:**
- Produces: `summarize_action_deltas(deltas: np.ndarray, bound: float) -> dict[str, float]`
- Produces: CLI option `--baseline-body-mass-scale FLOAT`
- Produces: optional `baseline_source`, `delta_shifted_source_minus_baseline_source`, and `residual_action_delta` JSON entries

- [ ] **Step 1: Write failing pure-summary and CLI tests**

Use a two-step, two-action delta array with exact expected mean absolute, RMS,
maximum, and near-bound fraction. Add empty, nonfinite, and nonpositive-bound
cases. Parse `--baseline-body-mass-scale 1.0` and assert the exact float.

- [ ] **Step 2: Run RED**

```bash
JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_tracking_residual_comparison
```

Expected: import or argument failures because the helper and CLI do not exist.

- [ ] **Step 3: Implement the smallest backward-compatible extension**

Add the pure summary helper. Append each bounded residual correction during
residual action construction. If the baseline option is present, construct the
nominal environment, run its source controller at every registered phase, and
add its per-phase and aggregate records. Add both registered delta maps and the
global action summary to the JSON. Preserve the old document when the option is
absent except for the additive residual-action summary.

- [ ] **Step 4: Run GREEN and adjacent regressions**

```bash
JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_tracking_residual_comparison \
  tests.test_g1_tracking_evaluator \
  tests.test_g1_tracking_runner
```

Expected: all tests pass.

- [ ] **Step 5: Verify, commit, and push**

Run `git diff --check`, inspect the complete diff, commit as
`Add G1 mass-shift headroom diagnostic`, and push the research code branch.

### Task 2: Registered Scientific Diagnostic

**Files:**
- Create: `research/experiments/E-20260805-087.yaml` in `diffsim2real-lab`
- Modify: `research/state/current.yaml`
- Modify after evaluation: `research/hypotheses/H-G1-024.yaml`
- Modify after evaluation: `docs/vault/findings/F-20260805-001 Residual SHAC Preserves the G1 Walking Basin but Nominal Gains Are Seed-Sensitive.md`

- [ ] **Step 1: Preregister exact provenance and outcomes**

Pin the executable SHA, nominal scale `1.0`, shifted scale `1.15`, exact
checkpoint SHA, phases, seed, solver, 30-minute walltime, complete JSON
artifact, and the design's four outcomes. Validate and dry-run.

- [ ] **Step 2: Commit and push registration**

Stage only E087 and current state, validate, commit, and push before execution.

- [ ] **Step 3: Execute exactly once**

```bash
uv run python -m tools.runexp E-20260805-087
```

Require return code zero, no timeout, complete finite JSON, and the fixed
checkpoint/model identities.

- [ ] **Step 4: Curate and select the next mechanism**

Apply the preregistered outcome. Update hypothesis, finding, current state, and
the isolated Obsidian projection; validate, commit, and push. If headroom is
insufficient, design one physically discriminative shift. If update-limited,
localize the residual objective/update. If recovered, proceed to the
collocation-transfer design.
