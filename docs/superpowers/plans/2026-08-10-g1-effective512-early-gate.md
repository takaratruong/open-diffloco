# G1 Effective-512 Early Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and execute an immutable canonical G1 SHAC runner whose two physical 256-environment gradient shards form one effective-512 update for 128 updates.

**Architecture:** Derive the complete configuration from `build_canonical_kwargs`, override only accumulation factor and its mechanically matched transition budget, and reuse the existing trainer accumulation implementation. Preregister the run in the research repository and select it only through the unchanged replay-free five-phase evaluation.

**Tech Stack:** Python 3.11, JAX/MJX, unittest, existing SHAC trainer, DiffSim2Real plain-file experiment registry.

## Global Constraints

- Physical differentiation width remains exactly 256 environments.
- Effective batch is exactly 512 environments from two independent shards.
- Horizon remains 12 and exactly 128 actor updates consume 786,432 transitions.
- Every canonical scientific setting other than effective batch and its derived transition budget remains unchanged.
- Direct 512-environment differentiation is not attempted because the established L40S resource gate OOMs.

---

### Task 1: Immutable effective-512 runner

**Files:**
- Create: `tests/test_g1_effective512_shac_runner.py`
- Create: `tools/run_g1_effective512_shac.py`

**Interfaces:**
- Consumes: `build_canonical_kwargs(profile_name, reference_path, seed)`.
- Produces: `build_effective512_kwargs(profile_name, reference_path, seed) -> dict` with factor two and 786,432 total steps.

- [ ] Write a test asserting that only `gradient_accumulation_steps` and `total_steps` differ from the canonical dictionary, while physical environments remain 256 and horizon remains 12.
- [ ] Run the test and confirm import failure because the runner does not exist.
- [ ] Implement the minimal immutable runner by following the existing horizon-24 wrapper structure.
- [ ] Run the focused runner and accumulation tests and confirm they pass.
- [ ] Commit the tested runner and its design documents.

### Task 2: Preregister and launch the bounded experiment

**Files:**
- Create in the research repository: `research/experiments/E-20260810-006.yaml`

**Interfaces:**
- Consumes: the exact clean code commit from Task 1 and the canonical G1 model/reference hashes.
- Produces: one guarded seed-zero run containing final/checkpoint policies, hparams, log, and diagnostics.

- [ ] Register one causal batch-size change, the 19/48 minimum/median advancement gate, pinned provenance, and a three-hour wall-time budget.
- [ ] Validate the registry and run an exact dry run.
- [ ] Launch through `python -m tools.runexp E-20260810-006` and verify the process reaches the guarded worker.

### Task 3: Evaluate and curate

**Files:**
- Modify after completion: `research/experiments/E-20260810-006.yaml`
- Modify after completion: `research/state/current.yaml`
- Modify after completion when warranted: `research/hypotheses/H-SCI-001.yaml`

**Interfaces:**
- Consumes: the exact final actor and immutable run artifacts.
- Produces: deterministic phase-grid metrics, trajectory evidence, a plot, and one registered outcome.

- [ ] Verify exact update count, effective population, finiteness, provenance, and every required training artifact.
- [ ] Evaluate exact phases 0/100/200/300/400 without replay or teleportation.
- [ ] Select only the preregistered advancement, finite-no-gain, or invalid-execution outcome.
- [ ] Validate the registry, rebuild the vault, run focused tests, and commit only owned research files.
