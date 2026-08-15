# E033 Recovery Action-Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parameterize the recovery correction cap and execute one exact 1.0-cap successor to E033.

**Architecture:** Keep the existing oracle intact and validate one runtime scalar. The environment retains its independent final normalized action clip.

**Tech Stack:** Python, argparse, JAX, Optax, pytest, YAML research registry.

## Global Constraints

- Default behavior remains correction bound 0.5.
- The treatment uses bound 1.0 and changes no other E033 setting.
- Final environment actions remain clipped to `[-1, 1]`.

---

### Task 1: Correction-bound interface

**Files:**
- Modify: `tools/run_g1_action_sequence_recovery_oracle.py`
- Test: `tests/test_g1_action_sequence_recovery_oracle.py`

**Interfaces:**
- Produces: `run_oracle(..., correction_bound: float = CORRECTION_BOUND)` and CLI `--correction-bound`.

- [ ] Write parser tests for default 0.5, custom 1.0, and rejection of zero, negative, NaN, and infinity.
- [ ] Run the focused test and observe the missing-interface failure.
- [ ] Thread the validated finite positive value through tanh scaling, execution validation, and summary telemetry.
- [ ] Run focused pytest, Ruff, py_compile, and diff-check.
- [ ] Commit only tool and test as `feat: parameterize recovery correction bound`.

### Task 2: E034 authority discriminator

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260815-034.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generate: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/experiments/E-20260815-034.md`

**Interfaces:**
- Consumes: Task 1 SHA and exact E033 assets/settings.
- Produces: one deterministic 24-start survival comparison.

- [ ] Register E034 with only `--correction-bound 1.0` as the causal delta.
- [ ] Validate and dry-run the registry command.
- [ ] Execute exactly once on one GPU and inspect every required artifact/gate.
- [ ] Curate the registered outcome, rebuild the vault, validate, and commit only owned files.
