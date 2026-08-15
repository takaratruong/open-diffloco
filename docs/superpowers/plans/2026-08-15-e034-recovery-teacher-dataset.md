# E034 Recovery Teacher Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a hash-bound state/action teacher dataset from the immutable E034 recovery tape.

**Architecture:** A focused evaluator reconstructs E034 and publishes raw replay tensors plus a pure validated summary. No training or generic orchestration is added.

**Tech Stack:** Python, NumPy, JAX/MJX, pytest, YAML research registry.

## Global Constraints

- Inputs and survival must reproduce E034 exactly.
- All 24 trajectories are retained; the success mask identifies thirteen usable teacher trajectories.
- No policy parameters, actions, states, or labels are modified.

---

### Task 1: Pure dataset evidence validator

**Files:**
- Create: `tools/build_g1_e034_recovery_teacher_dataset.py`
- Create: `tests/test_g1_e034_recovery_teacher_dataset.py`

- [ ] Write failing tests for exact shapes, survival, clip identity, success count, finite values, and clip summaries.
- [ ] Run focused pytest and observe the missing-module failure.
- [ ] Implement the minimal pure validation and atomic publication helpers.
- [ ] Run focused pytest, Ruff, py_compile, and diff-check.
- [ ] Commit as `feat: validate E034 recovery teacher data`.

### Task 2: Exact E034 replay collector

**Files:**
- Modify: `tools/build_g1_e034_recovery_teacher_dataset.py`
- Modify: `tests/test_g1_e034_recovery_teacher_dataset.py`

- [ ] Add provenance-validated replay using existing E023 environment/policy/state constructors.
- [ ] Capture the exact registered tensor contract before each environment step.
- [ ] Add parser/provenance tests and run the focused suite and static gates.
- [ ] Commit as `feat: collect E034 recovery teacher data`.

### Task 3: Register and curate E035

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260815-035.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generate: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/experiments/E-20260815-035.md`

- [ ] Register the exact replay/data-publication experiment and its invalid branch.
- [ ] Validate, dry-run, commit registration, and execute once on one GPU.
- [ ] Verify artifact hashes and exact replay gates, curate, rebuild the vault, and commit only owned files.
