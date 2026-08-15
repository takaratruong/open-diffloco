# E032 Worst-Margin Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact update-budget control to the recovery oracle and run one preregistered 256-update convergence discriminator.

**Architecture:** Preserve the existing oracle and expose only its update count as a validated argument. The experiment registry owns the scientific delta and deterministic result curation.

**Tech Stack:** Python, argparse, JAX, Optax, pytest, YAML research registry.

## Global Constraints

- Default oracle behavior remains exactly 64 updates.
- E033 changes only the update count from 64 to 256 relative to E032.
- H32, independent tapes, worst-margin objective, correction bound 0.5, Adam LR 0.03, per-start gradient clip 1.0, source bank, policy, solver, seed, and environment remain exact.
- No parameter sweep, policy training, no-reset environment change, or action-authority change is in scope.

---

### Task 1: Validated update-budget interface

**Files:**
- Modify: `tools/run_g1_action_sequence_recovery_oracle.py`
- Test: `tests/test_g1_action_sequence_recovery_oracle.py`

**Interfaces:**
- Consumes: existing `run_oracle(...)` and `build_parser()`.
- Produces: `run_oracle(..., updates: int = UPDATES)` and CLI `--updates`.

- [ ] **Step 1: Write failing tests**

Add tests asserting the parser defaults to 64, accepts `--updates 256`, rejects
zero/negative values, and source inspection confirms the loop and summary use
the validated runtime value rather than the module constant.

- [ ] **Step 2: Verify RED**

Run:
`/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_action_sequence_recovery_oracle.py`

Expected: failures because `--updates` and the runtime parameter do not exist.

- [ ] **Step 3: Implement the minimal interface**

Add a positive-integer argparse converter, thread `updates` into `run_oracle`,
iterate `range(1, updates + 1)`, and persist `"updates": updates`.

- [ ] **Step 4: Verify GREEN and static gates**

Run the focused pytest command, Ruff on the two files, `py_compile` on the tool,
and `git diff --check`. Expected: all pass.

- [ ] **Step 5: Commit**

Commit only the tool and focused test with message
`feat: parameterize recovery oracle budget`.

### Task 2: Register, execute, and curate E033

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260815-033.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generate: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/experiments/E-20260815-033.md`

**Interfaces:**
- Consumes: Task 1 commit and exact E032 provenance/assets.
- Produces: deterministic E033 summary/NPZ and curated registry outcome.

- [ ] **Step 1: Register one causal experiment**

Copy E032's contract, set parent E032 and the Task 1 code SHA, add `--updates
256`, and specify convergence/recovery/invalid outcome branches.

- [ ] **Step 2: Validate and dry-run**

Run `uv run python tools/researchctl.py validate` and
`uv run python -m tools.runexp E-20260815-033 --dry-run`.

- [ ] **Step 3: Execute once**

Run `uv run python -m tools.runexp E-20260815-033`, inspect the final summary and
NPZ against every registered gate, and do not launch another experiment before
curation.

- [ ] **Step 4: Curate evidence**

Select only the preregistered outcome, update `current.yaml`, rebuild the vault,
validate the registry, and stage only E033/current/generated E033 page.

- [ ] **Step 5: Commit**

Commit the exact owned registry projection with message
`research: curate E033 oracle convergence`.
