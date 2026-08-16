# G1 Objective-Direction Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute one frozen E023 diagnostic that distinguishes stochastic-gradient variance, H24/H48 credit mismatch, terminal-bootstrap usefulness, and aligned-but-behaviorally-unsafe optimization.

**Architecture:** Keep reusable numerical contracts in one small SHAC module and put E023/LAFAN provenance, environment reconstruction, gradient capture, proposal rollout, and manifest-last publication in one dedicated experiment tool. Reuse the existing G1 environment/policy loaders, residual adapter, CAGrad reducer, carried-bank schema, and compiled phase-grid conventions; do not modify the ordinary SHAC training path.

**Tech Stack:** Python 3.11, JAX x64, Flax, NumPy, MJX, pytest, Ruff, existing `g1-4x5` solver and DiffSim registry.

## Global Constraints

- One frozen diagnostic only; no policy training, activation change, hyperparameter sweep, or retained candidate.
- Exact E023 checkpoint SHA-256 `2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f`.
- Exact E023 hparams SHA-256 `a4435aebb4be1d3f539fb82634b47134424a57726fc11c4f0011821bc15ff650`.
- Exact LAFAN reference SHA-256 `bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db`.
- Exact 120-row carried bank SHA-256 `d91dfb1b5190f14a5204cb16abbf527ede4f08e0a9b46cec9dfa602500d708a5`.
- Exact model/controller SHA-256 `5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1` / `f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee`.
- Fixed state population 512, fixed float32 noise tapes `(512,48,29)`, horizons 24 and 48, five phase bins, per-environment clip 1.0, CAGrad alpha 0.5 for 32 iterations.
- Equal-norm base displacement `0.09495018422603607`; multipliers exactly `0.125`, `0.25`, `0.5`, `1.0`.
- All outputs are atomic and hash-bound; JSON completion manifest is written last.

---

### Task 1: Pure gradient and outcome contracts

**Files:**
- Create: `src/algorithms/shac/objective_direction_audit.py`
- Create: `tests/test_shac_objective_direction_audit.py`

**Interfaces:**
- Consumes: existing `accumulate_phase_gradients`, `finalize_phase_gradients`, and `combine_cagrad` from `src.algorithms.shac.cagrad`.
- Produces: `tree_l2_norm(tree)`, `tree_dot(first, second)`, `tree_cosine(first, second)`, `normalized_descent_proposal(params, gradient, displacement)`, `aggregate_audit_direction(per_env_gradients, phases, *, phase_count, clip_norm, alpha, iterations)`, `select_carried_safe_candidate(rows, *, baseline_survival)`, and `classify_objective_direction_audit(...)`.

- [ ] **Step 1: Write failing pure-contract tests**

  Cover exact cosine for aligned/opposed/orthogonal vectors, zero-vector rejection, equal proposal displacement, clip-before-bin aggregation, five occupied bins, carried-safe lexicographic ordering, ordinary componentwise preservation, outcome precedence, and non-finite rejection.

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_objective_direction_audit.py
  ```

  Expected: collection fails because `src.algorithms.shac.objective_direction_audit` does not exist.

- [ ] **Step 3: Implement minimal pure module**

  Flatten only numeric PyTree leaves in deterministic JAX tree order. Reject empty, zero-norm, shape-mismatched, or non-finite directions. Use existing CAGrad primitives rather than reimplementing the solver. Return JSON-safe diagnostic scalars separately from gradient PyTrees.

- [ ] **Step 4: Run GREEN and neighbors**

  ```bash
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/test_shac_objective_direction_audit.py \
    tests/test_shac_cagrad.py \
    tests/test_shac_gradient_aggregation.py
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check \
    src/algorithms/shac/objective_direction_audit.py \
    tests/test_shac_objective_direction_audit.py
  ```

  Expected: all selected tests pass and Ruff exits zero.

- [ ] **Step 5: Commit**

  ```bash
  git add src/algorithms/shac/objective_direction_audit.py tests/test_shac_objective_direction_audit.py
  git commit -m "feat: add objective direction audit contracts"
  ```

### Task 2: Exact frozen G1 gradient capture

**Files:**
- Create: `tools/evaluate_g1_objective_directions.py`
- Create: `tests/test_g1_objective_direction_audit.py`

**Interfaces:**
- Consumes: Task 1 contracts; existing `_build_environment`, `_load_policy`, `_load_all_bank_rows`, `validate_e023_hparams`, `validate_runtime_assets`, `transplant_zero_head_recovery_features`, `apply_frozen_preview_residual`, and atomic file/hash helpers.
- Produces: `build_fixed_population_indices(seed)`, `build_fixed_noise_tapes(seed)`, `validate_common_noise_prefix(tape_a, tape_b)`, `build_preflight(...)`, `run_gradient_capture(...)`, `validate_gradient_artifacts(...)`, and CLI `python -m tools.evaluate_g1_objective_directions`.

- [ ] **Step 1: Write failing provenance, state, and artifact tests**

  Tests require exact hashes and seed zero; deterministic 128/384 source counts; all 120 carried rows represented before eight deterministic repeats; exact H24/H48 tape-A prefix; independent tape B; E041 zero-head action identity; finite complete four-direction summaries; nonzero phase counts; aggregate vector/summary hash agreement; and manifest-last tamper rejection.

- [ ] **Step 2: Run RED**

  ```bash
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_objective_direction_audit.py
  ```

  Expected: collection fails because `tools.evaluate_g1_objective_directions` does not exist.

- [ ] **Step 3: Implement exact loader and fixed batch**

  Reuse the validated E038/E041 loader seams. Restore carried qpos/qvel/phase/last action/full history and construct exact reference states under nominal randomization. Persist source kind/index, phase, qpos/qvel/history identities, RNG keys, and tape hashes. Reconstruct the E041 transferred hidden layer with a byte-zero output head and require bit-exact E023 composite actions on the complete fixed batch.

- [ ] **Step 4: Implement component losses and capture**

  Define one rollout kernel parameterized by static horizon and bootstrap mode. It must reproduce training's normalized observations, frozen-parent residual composition, RMR reparameterized action noise, sampled-action clip, environment step, discounted done/terminal accumulation, anchor term, and target-critic evaluation. Compute per-environment adapter gradients for H24-A, H24-B, H48-A, and bootstrap-A; aggregate through Task 1; materialize only aggregate/phase vectors and per-environment norm/cosine summaries.

- [ ] **Step 5: Run GREEN plus compiled micro-smoke**

  ```bash
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/test_g1_objective_direction_audit.py \
    tests/test_shac_objective_direction_audit.py \
    tests/test_g1_e038_recovery_transfer.py \
    tests/test_g1_zero_head_feature_transfer.py
  CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=true PYTHONPATH=. \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m \
    tools.evaluate_g1_objective_directions --smoke
  ```

  Expected: focused tests pass; the one-environment H24/H48/bootstrap smoke publishes a non-scientific valid smoke summary with finite nonzero adapter gradients and exact common-prefix noise.

- [ ] **Step 6: Commit**

  ```bash
  git add tools/evaluate_g1_objective_directions.py tests/test_g1_objective_direction_audit.py
  git commit -m "feat: capture frozen G1 objective directions"
  ```

### Task 3: Bounded proposal evaluation and classification

**Files:**
- Modify: `tools/evaluate_g1_objective_directions.py`
- Modify: `tests/test_g1_objective_direction_audit.py`

**Interfaces:**
- Consumes: validated aggregate H24-A/H48-A/bootstrap-A vectors from Task 2 and Task 1 proposal/selection contracts.
- Produces: `evaluate_carried_proposals(...)`, `evaluate_ordinary_candidate(...)`, `publish_completion_manifest(...)`, plot artifacts, and final ordered `selection.json`.

- [ ] **Step 1: Add failing end-to-end artifact tests**

  Test twelve exact proposal labels, equal parameter displacement, complete 120-row H32 evidence, carried-safe preselection, at-most-one ordinary candidate per direction, exact five-phase baseline, no retained checkpoint field, ordered outcome classification, plot/hash presence, partial-publication rejection, and raw NPZ tamper rejection.

- [ ] **Step 2: Run RED**

  ```bash
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/test_g1_objective_direction_audit.py -k 'proposal or publication or outcome'
  ```

  Expected: tests fail because proposal evaluation and completion publication are absent.

- [ ] **Step 3: Implement carried and ordinary rollouts**

  Reuse the existing compiled frozen-preview rollout boundary. Evaluate all proposals and baseline on the bank with no noise. Select at most one candidate per direction using the pure contract; evaluate those candidates on phases `0/100/200/300/400`; never save a deployable checkpoint or label a candidate selected policy.

- [ ] **Step 4: Implement plots and manifest-last publication**

  Plot the aggregate/per-phase cosine matrix and carried/ordinary survival deltas. Write NPZ and plots atomically, hash them into JSON summaries, independently reopen and validate every artifact, then write the completion manifest last.

- [ ] **Step 5: Run GREEN and static checks**

  ```bash
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/test_g1_objective_direction_audit.py \
    tests/test_shac_objective_direction_audit.py \
    tests/test_g1_e038_recovery_transfer.py \
    tests/test_g1_zero_head_feature_transfer.py \
    tests/test_g1_conflict_projected_recovery_teacher_selector.py
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check \
    src/algorithms/shac/objective_direction_audit.py \
    tools/evaluate_g1_objective_directions.py \
    tests/test_shac_objective_direction_audit.py \
    tests/test_g1_objective_direction_audit.py
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
    src/algorithms/shac/objective_direction_audit.py \
    tools/evaluate_g1_objective_directions.py
  git diff --check
  ```

  Expected: every command exits zero.

- [ ] **Step 6: Commit**

  ```bash
  git add tools/evaluate_g1_objective_directions.py tests/test_g1_objective_direction_audit.py
  git commit -m "feat: evaluate G1 objective direction proposals"
  ```

### Task 4: Register, run, evaluate, and curate E-20260816-000

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260816-000.yaml`
- Modify after validated completion: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Modify after validated completion: the directly supported finding under `/home/ubuntu/projects/diffsim2real-lab/docs/vault/findings/`
- Generate after validated completion: Dashboard and experiment projection with `tools/researchctl.py build-vault`

**Interfaces:**
- Consumes: clean implementation commit from Tasks 1-3 and the exact approved design outcome map.
- Produces: one guarded run under `$DIFFSIM_SEED_DIR`, a validated registry result, updated current decision, finding edge when promotion criteria hold, and Dashboard plots.

- [ ] **Step 1: Verify the implementation boundary**

  Run the complete focused suite and static checks from Task 3, confirm clean Git status, record HEAD, compute dirty-patch SHA-256, and push the branch to `takaratruong/open-diffloco`.

- [ ] **Step 2: Register exactly one approved experiment**

  Write schema-v1 E-20260816-000 with one diagnostic causal change, all eight ordered outcomes, exact input/code/environment/model/reference/controller/solver hashes, seed zero, one GPU, 180-minute wall-time budget, and command output rooted at `$DIFFSIM_SEED_DIR`.

- [ ] **Step 3: Validate and dry-run**

  ```bash
  uv run python tools/researchctl.py validate
  uv run python -m tools.runexp E-20260816-000 --dry-run
  ```

  Expected: registry valid and dry-run resolves the exact command without execution.

- [ ] **Step 4: Launch and monitor the sole run**

  ```bash
  uv run python -m tools.runexp E-20260816-000
  ```

  Do not modify scientific code or inputs while active. Monitor logs and GPU state; stop only on registered invalidity or timeout.

- [ ] **Step 5: Independently validate and curate**

  Reopen the completion manifest, recompute every SHA-256, rerun the pure classifier, and require exact agreement. Update the experiment result/current state/finding only from that evidence. Copy plots into the untracked Dashboard preview path, rebuild the vault, validate the registry, and run focused registry tests.

- [ ] **Step 6: Commit and push owned research files**

  Stage only E-20260816-000, current state, the qualifying finding, generated E-20260816-000 page, Dashboard, and preview manifest if tracked by existing convention. Preserve every unrelated dirty vault file. Commit with `research: conclude E-20260816-000 objective audit` and push lab `main`.

---

## Plan self-review

- Spec coverage: all immutable inputs, four direction captures, common-noise control, equal-norm proposal gate, ordered outcomes, artifacts, and stop rule map to Tasks 1-4.
- Placeholder scan: no TBD/TODO or unspecified implementation step remains.
- Type consistency: Task 2/3 consume only Task 1's named pure interfaces; Task 4 consumes the single Task 3 CLI and manifest.
- Scope: one reusable pure module, one experiment tool, two focused test modules, and one registry record; no training-path change or generic framework.
