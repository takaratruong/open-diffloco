# Canonical G1 Detached Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the canonical fixed-4x5 G1 SHAC run from checkpoint 2,359,296 without changing its scientific contract and keep it alive across Codex restarts.

**Architecture:** Add one optional operational path to the existing canonical kwargs boundary and CLI, reusing the already-tested `train(resume_from=...)` implementation. Register a successor experiment and execute its guarded command in detached `tmux`.

**Tech Stack:** Python 3.11, argparse, unittest, JAX/MJX, tmux, plain-file research registry.

## Global Constraints

- Change no canonical scientific hyperparameter.
- Resume only checkpoint SHA-256 `1afd0785cacfbf19cfb5c17d6207561a536a7952fe92abe725371e5e0600d584`.
- Use solver profile `g1-4x5`, seed 0, and total target 8,000,000 transitions.
- Keep the source E-20260810-002 artifacts immutable and write successor output separately.

---

### Task 1: Transport the exact resume checkpoint

**Files:**
- Modify: `tests/test_canonical_g1_shac_runner.py`
- Modify: `tools/run_canonical_g1_shac.py`

**Interfaces:**
- Consumes: `train(resume_from: str, **canonical_kwargs)` from `src.algorithms.shac.algorithm`.
- Produces: `build_canonical_kwargs(..., resume_from: str | Path | None = None)` and CLI `--resume-from PATH`.

- [ ] **Step 1: Write the failing test**

Add a test that calls `build_canonical_kwargs("g1-4x5", reference, 0, resume_from=checkpoint)` and asserts `kwargs["resume_from"] == str(checkpoint.resolve())`, while a call without the option contains no `resume_from` key.

- [ ] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=True /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_canonical_g1_shac_runner.py -q`

Expected: failure because `build_canonical_kwargs` does not accept `resume_from`.

- [ ] **Step 3: Implement the minimal transport**

Add the optional builder parameter, resolve and add it only when non-null, add `parser.add_argument("--resume-from", type=Path)`, and pass it from `main`.

- [ ] **Step 4: Verify GREEN and neighboring resume behavior**

Run: `JAX_ENABLE_X64=True /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_canonical_g1_shac_runner.py tests/test_shac_exact_resume.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

Commit only the runner, tests, design, and plan; push the research branch.

### Task 2: Register and launch the detached successor

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260810-003.yaml`
- Modify through generator: the corresponding Obsidian experiment/current views.

**Interfaces:**
- Consumes: the pushed canonical runner commit and source checkpoint SHA-256.
- Produces: one approved guarded experiment and one detached `tmux` session.

- [ ] **Step 1: Recompute and pin provenance**

Hash the checkpoint, model, reference, clean code commit, and dirty patch; reject launch if any identity differs from the registered successor.

- [ ] **Step 2: Write and validate the successor record**

Set the sole causal change to exact continuation plus detached execution. Require finite restoration at step 2,359,296, complete 8M final artifacts, and the registered phase grid.

- [ ] **Step 3: Dry-run the guarded experiment**

Run: `uv run python -m tools.runexp E-20260810-003 --dry-run`

Expected: approved contract resolves with no execution.

- [ ] **Step 4: Launch detached and verify liveness**

Start `uv run python -m tools.runexp E-20260810-003` in a named detached tmux session. Verify the session, child process, GPU allocation, copied contract, and initial log showing checkpoint restoration.

- [ ] **Step 5: Update and push the research graph**

Run registry validation and vault generation, commit only owned registry/projection files, and push main.
