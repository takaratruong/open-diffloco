# E038 Recovery Expert Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the frozen E038 recovery expert against its frozen E023 parent on all 120 immutable E027 carried states before any SHAC update.

**Architecture:** Add one experiment-specific evaluator that validates immutable inputs, reconstructs all bank states, runs paired H32 MJX scans, classifies the registered transfer outcome, and atomically publishes tensor evidence followed by a hash-bound summary. Reuse the existing environment, policy loader, residual adapter, solver profile, and artifact helpers.

**Tech Stack:** Python, JAX, MJX, Flax, NumPy, pytest, Ruff, the existing DiffSim experiment registry.

## Global Constraints

- Exact E023 checkpoint/hparams, E038 expert, E027 bank, LAFAN reference, model, controller, seed zero, and `g1-4x5` solver.
- No training, parameter update, randomization, observation noise, reset noise, pushes, assistance, or action-boundary change.
- All 120 states and both paired arms must be finite and artifact-bound.
- Production behavior is written test-first.

---

### Task 1: Pure transfer contracts

**Files:**
- Create: `tests/test_g1_e038_recovery_transfer.py`
- Create: `tools/evaluate_g1_e038_recovery_transfer.py`

**Interfaces:**
- Produces: `validate_bank_layout(arrays) -> dict[str, np.ndarray]`
- Produces: `survival_from_terminals(terminals) -> list[int]`
- Produces: `classify_transfer(parent_survival, expert_survival, source_start_phase, execution_valid) -> str`

- [ ] **Step 1: Write failing tests** for the exact 120-row/5x24 bank layout, malformed source groups, survival extraction, every registered outcome branch, and seed-zero enforcement.
- [ ] **Step 2: Run RED** with `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_e038_recovery_transfer.py`; expect missing-module failures.
- [ ] **Step 3: Implement the minimal pure helpers** and constants in the evaluator.
- [ ] **Step 4: Run GREEN** and require all focused tests to pass.
- [ ] **Step 5: Commit** only the evaluator and test module.

### Task 2: Paired MJX evaluation and artifacts

**Files:**
- Modify: `tests/test_g1_e038_recovery_transfer.py`
- Modify: `tools/evaluate_g1_e038_recovery_transfer.py`

**Interfaces:**
- Consumes: E023 `_load_policy`, E038 `PreviewResidualAdapter` parameters, E027 bank arrays.
- Produces: `run_evaluation(...) -> dict[str, object]`, `paired_rollouts.npz`, and `summary.json`.

- [ ] **Step 1: Add failing artifact tests** requiring exact paired tensor names/shapes, bit-exact `effective_action == clip(raw_action,-1,1)`, finite values, exact source grouping, manifest-last hashes, and parameter immutability.
- [ ] **Step 2: Run RED** and confirm failures are from missing evaluation/publication behavior.
- [ ] **Step 3: Implement paired state reconstruction and H32 scans** using the exact environment, frozen normalizer, parent actor, E038 expert, and `g1-4x5` context. Record qpos and all registered tensors.
- [ ] **Step 4: Implement atomic NPZ/JSON publication** and bind every input/output hash plus outcome metrics.
- [ ] **Step 5: Run focused GREEN, Ruff, py_compile, and `git diff --check`.**
- [ ] **Step 6: Commit** the bounded evaluator implementation.

### Task 3: Review, register, execute, and curate E039

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260815-039.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generated: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/experiments/E-20260815-039.md`
- Possibly modify: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/findings/F-20260815-001 Recovery Oracle Authority and Distillation Boundary.md`

**Interfaces:**
- Consumes: clean reviewed evaluator commit.
- Produces: one curated E039 result that fixes the subsequent SHAC initialization/anchor design.

- [ ] **Step 1: Request bounded independent review** of provenance, paired-state equality, frozen parameters, action boundary, outcome precedence, and artifact validation; repair Important findings test-first.
- [ ] **Step 2: Register E039** with the exact code commit, hashes, seed, GPU, command, 15-minute budget, outcome map, and required artifacts.
- [ ] **Step 3: Run registry validation and `tools.runexp E-20260815-039 --dry-run`; approve only after both pass.**
- [ ] **Step 4: Execute exactly once** with `uv run python -m tools.runexp E-20260815-039` and monitor the registered process.
- [ ] **Step 5: Independently validate artifacts and select only a preregistered outcome.**
- [ ] **Step 6: Curate registry/current/finding, rebuild the vault, validate, and commit only owned files.**
