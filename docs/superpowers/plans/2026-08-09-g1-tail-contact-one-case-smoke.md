# G1 Tail-Contact One-Case Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one authoritative G1 tail-contact derivative smoke that immediately decides whether to build the real forward-mode SHAC update.

**Architecture:** Add a focused smoke runtime beside the existing pure E012 components. It streams only E011 shard 0, validates the selected row against authoritative receipts, compiles reusable reverse and directional-JVP kernels, writes one atomic operational receipt outside the E012 run tree, and applies the two-branch decision rule.

**Tech Stack:** Python, JAX/MJX, NumPy, existing E064/E011 audit components, pytest, Ruff.

## Global Constraints

- Exactly one case: shard 0, phase bin 0.
- No policy, optimizer, state, or training update.
- No automatic twenty-case E012 launch.
- Sequential 29-direction JVP assembly; never tangent-vmap.
- Output is operational telemetry outside `runs/E-20260809-012`.
- Forward-valid plus resource-valid authorizes an actual SHAC method experiment; any failure abandons this mechanism without further diagnostic variants.

---

### Task 1: Compiled one-case runtime

**Files:**
- Create: `src/algorithms/shac/g1_tail_contact_derivative_smoke.py`
- Create: `tests/test_g1_tail_contact_derivative_smoke.py`

**Interfaces:**
- Consumes: `FirstActionObjective`, `DerivativeComparison`, canonical direction seed 12001.
- Produces: `CompiledCaseSmoke`, `compile_case_kernels(...)`, and `run_compiled_case_smoke(...)` with separate compile and blocked cached timings.

- [ ] Write failing tests for one reverse compile/cached execution, one directional-JVP compile, three sequential 29-call cached sweeps, two probes, timing fields, forward gates, and rejection of noncanonical inputs.
- [ ] Run `JAX_ENABLE_X64=true python -m pytest -q tests/test_g1_tail_contact_derivative_smoke.py` and confirm missing-module/API failures.
- [ ] Implement the minimal compiled runtime, blocking every measured result before stopping its timer.
- [ ] Re-run the focused tests and Ruff.
- [ ] Commit the task.

### Task 2: Authoritative smoke orchestration and receipt

**Files:**
- Modify: `src/algorithms/shac/g1_tail_contact_derivative_smoke.py`
- Create: `tools/smoke_g1_tail_contact_derivatives.py`
- Modify: `tests/test_g1_tail_contact_derivative_smoke.py`
- Create: `tests/test_g1_tail_contact_derivative_smoke_cli.py`

**Interfaces:**
- Consumes: verified base audit contract, verified E011 run directory, `_prepare_e064_execution`, `load_e011_source_receipts`, `make_frozen_action_noise`, and Task 1 runtime.
- Produces: one atomic `smoke_receipt.json` and process exit zero only for a complete classified smoke.

- [ ] Write failing tests for exact shard-0 receipt reproduction, bin-0 rank selection, authoritative trajectory injection, manifest-authenticated source, atomic output, refusal to overwrite, and absence of E012 run artifacts.
- [ ] Run the new tests and confirm the expected failures.
- [ ] Implement streaming shard-0 orchestration and receipt publication.
- [ ] Add host peak-RSS and best-effort JAX device memory snapshots without making unavailable allocator stats an execution failure.
- [ ] Re-run focused and adjacent audit tests plus Ruff and `py_compile`.
- [ ] Commit the task.

### Task 3: Independent review and live smoke

**Files:**
- Modify only if review finds a demonstrated defect, with a failing regression test first.

**Interfaces:**
- Consumes: clean Task 1/2 commit.
- Produces: pushed implementation commit and one immutable operational smoke directory.

- [ ] Review the complete diff against the design, specifically JIT compilation accounting, exact E011 replay, sequential execution, memory bounds, and output isolation.
- [ ] Run the full focused suite, Ruff, `py_compile`, and `git diff --check` from a clean worktree.
- [ ] Push the clean implementation commit.
- [ ] Launch the smoke in a fresh GPU process with the pinned E064/E011 inputs.
- [ ] Verify receipt hashes, decision gates, process return code, and absence of E012 evidence.
- [ ] If authorized, preregister the actual forward-JVP plus policy-VJP SHAC update; otherwise record abandonment and pivot without another derivative diagnostic.
