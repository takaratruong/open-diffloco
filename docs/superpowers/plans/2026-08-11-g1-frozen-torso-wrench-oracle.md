# Frozen E008 Torso-Wrench Oracle Implementation Plan

> Execute in the isolated `research/g1-backpack-oracle-20260811` worktree. Do
> not modify or consume E010 artifacts.

**Goal:** Determine whether a bounded analytic torso wrench lets the frozen E008
actor complete all five strict reference suffixes.

**Architecture:** Add a small evaluation-only pure wrench module and a dedicated
paired evaluator. The evaluator injects `xfrc_applied` into the state before the
unchanged environment step. Production training paths remain untouched.

---

### Task 1: Pure bounded torso-wrench controller

**Files:**
- Create: `src/evaluation/g1_torso_wrench_oracle.py`
- Create: `tests/test_g1_torso_wrench_oracle.py`

Write failing tests first for zero scale, shortest quaternion error, yaw/world
rotation, exact force/torque channels, overwrite behavior, norm caps, and finite
outputs. Implement the smallest typed pure functions needed to make them pass.
Use environment-provided nominal mass/gravity and resolve the torso body by name
rather than relying only on the documented numeric ID.

Run:

```bash
conda run -n diffsim python -m pytest tests/test_g1_torso_wrench_oracle.py -q
```

### Task 2: Paired strict phase-grid evaluator

**Files:**
- Create: `tools/evaluate_g1_frozen_torso_wrench_oracle.py`
- Create: `tests/test_g1_frozen_torso_wrench_oracle.py`

Write failing tests first for parser defaults, identical reset construction,
disabled exact-zero injection, per-phase wrench telemetry, immutable provenance,
and the exact `499/399/299/199/99` pass gate. Reuse existing actor loading,
normalization, phase-grid summary, solver context, and tracking metrics instead
of duplicating them. Write JSON atomically.

Run:

```bash
conda run -n diffsim python -m pytest \
  tests/test_g1_torso_wrench_oracle.py \
  tests/test_g1_frozen_torso_wrench_oracle.py \
  tests/test_g1_tracking_evaluator.py -q
```

### Task 3: Review and register the experiment

Run Ruff, `py_compile`, `git diff --check`, focused tests, and an independent
code review. Commit only the scoped evaluator, tests, and documents. Register a
new experiment in the lab repository with exact code/checkpoint/reference SHA,
fixed gains/caps, solver, seed, phases, and output paths before using a GPU.

### Task 4: Execute and decide

Run the paired evaluator on a free GPU. Preserve JSON evidence, raw wrench
traces, and rendered assisted/unassisted videos. Verify caps and finite values
independently from the payload. Report the five assisted and unassisted survival
counts. Build no learned assistance unless the assisted counts are exactly
`499/399/299/199/99`.
