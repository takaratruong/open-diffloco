# G1 Zero-Bootstrap Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one exact E012 continuation that isolates the causal effect of removing actor terminal-value bootstrapping.

**Architecture:** Add a narrow resume resolver to SHAC and an experiment-specific immutable runner derived from the verified zero-assistance continuation. The runner changes only the endpoint and bootstrap scale, validates all dense checkpoints, and leaves evaluation to the established phase-grid tools.

**Tech Stack:** Python, JAX, pytest, Ruff.

## Global Constraints

- Exact E012 checkpoint/hparams and all non-treatment settings are immutable.
- Exactly 32 effective-512 H12 updates and 196,608 transitions are allowed.
- Torso assistance is exact zero for every update.
- No critic, actor architecture, reset, noise, randomization, solver, or reward change is allowed.

---

### Task 1: Explicit bootstrap resume authority

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Test: `tests/test_shac_bootstrap_schedule.py`

**Interfaces:**
- Produces: `resolve_actor_bootstrap_resume_scale(resumed_hparams, *, requested_scale, allow_change) -> float` and train argument `allow_resume_actor_bootstrap_scale_change: bool = False`.

- [ ] Write tests that legacy/matching scales restore, unauthorized mismatch rejects, authorized zero is retained, and bool/nonfinite/negative inputs reject.
- [ ] Run RED and confirm missing-interface failures.
- [ ] Implement the resolver, call it before generic resume restoration, remove the unconditional overwrite, and persist the authority flag in hparams.
- [ ] Run GREEN plus bootstrap and resume-neighbor tests.

### Task 2: Immutable 32-update zero-bootstrap runner

**Files:**
- Create: `tools/run_g1_zero_bootstrap_continuation.py`
- Create: `tests/test_g1_zero_bootstrap_continuation_runner.py`

**Interfaces:**
- Consumes: exact E012 final checkpoint and `build_frozen_residual_assistance_kwargs`.
- Produces: fixed kwargs, preflight JSON, four dense checkpoints, and training-validation JSON.

- [ ] Write tests proving the only parent-kwargs deltas are endpoint, scale zero, and explicit resume authority; parser has no scientific overrides; artifact validation requires scale zero, exact-zero assistance, frozen parent, finite residual updates, and valid five-bin CAGrad.
- [ ] Run RED, implement the minimal runner, then run GREEN.
- [ ] Run focused tests, Ruff, `py_compile`, and `git diff --check`; commit and request bounded review.

### Task 3: Register, execute, evaluate, and curate

**Files:**
- Create in lab: `research/experiments/E-20260812-008.yaml`
- Modify after run: experiment result, current state, hypothesis projection, and generated vault notes.

- [ ] Pin reviewed clean code and exact inputs, validate registry, dry-run, and approve under standing simulation authority.
- [ ] Execute the guarded continuation on one free L40S.
- [ ] Evaluate every registered checkpoint with the existing replay-free five-phase evaluator and select only by the preregistered minimum/median/mean/earliest rule.
- [ ] Curate the exact outcome, rebuild the vault, verify, and commit only owned files.
