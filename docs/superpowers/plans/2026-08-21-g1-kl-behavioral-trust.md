# G1 KL-Calibrated Behavioral Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether one ordinary E023 SHAC proposal contains a PPO-proximal, componentwise-safe walking update before implementing a full adaptive-KL trainer.

**Architecture:** A one-update runner produces an immutable ordinary SHAC proposal. A separate evaluator builds six evaluation-only actor interpolations, measures diagonal-Gaussian KL and empirical log ratios on the frozen E023 audit bank, evaluates all candidates on the exact E023 replay-free phase grid, and applies a fail-closed selector. Normal training code remains unchanged.

**Tech Stack:** Python 3.11, JAX/Flax, NumPy, MJX, pytest, Ruff, existing G1 SHAC runners and phase-grid evaluator.

## Global Constraints

- Parent checkpoint is exact E023 update 128 at transition `1_572_864`.
- Proposal is exactly one effective-512 H24 five-bin CAGrad update ending at `1_585_152`.
- Candidate alphas are exactly `(1, 0.5, 0.25, 0.125, 0.0625, 0.03125)`.
- Mean per-dimension KL must be at most `0.01`; per-state p95 KL must be at most `0.02`.
- All candidate checkpoints are compact actor-plus-source-normalizer records marked evaluation-only and non-resumable.
- Behavioral floors are exactly `(116, 99, 67, 49, 24)` at phases `(0, 25, 50, 75, 100)`.
- Do not alter SHAC, the reward, the environment, the solver, the state distribution, or the E023 action-noise schedule.
- Use TDD: observe each focused test fail for the intended missing behavior before production edits.

---

### Task 1: Pure KL, interpolation, and selection contracts

**Files:**
- Create: `tools/evaluate_g1_e023_kl_behavioral_trust.py`
- Create: `tests/test_g1_e023_kl_behavioral_trust.py`

**Interfaces:**
- Produces: `interpolate_actor_tree(source, proposal, alpha) -> Any`
- Produces: `gaussian_proximity(old_mean, new_mean, std, epsilon) -> dict[str, Any]`
- Produces: `select_kl_behavioral_candidate(rows, source_survival=E023_FLOORS) -> dict[str, Any]`
- Consumes later: candidate builder and experiment evaluator use these exact functions and constants.

- [ ] **Step 1: Write interpolation RED tests**

```python
def test_interpolate_actor_tree_preserves_endpoints_and_rejects_drift():
    source = {"w": np.array([0.0, 2.0], np.float32)}
    proposal = {"w": np.array([2.0, 4.0], np.float32)}
    assert np.array_equal(interpolate_actor_tree(source, proposal, 0.5)["w"], [1, 3])
    assert np.array_equal(interpolate_actor_tree(source, proposal, 1.0)["w"], proposal["w"])
    with pytest.raises(ValueError):
        interpolate_actor_tree(source, {"x": proposal["w"]}, 0.5)
```

- [ ] **Step 2: Run interpolation RED**

Run: `JAX_PLATFORMS=cpu /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_e023_kl_behavioral_trust.py`

Expected: FAIL because `tools.evaluate_g1_e023_kl_behavioral_trust` does not exist.

- [ ] **Step 3: Implement strict tree interpolation**

Implement structural equality with `jax.tree_util.tree_structure`, finite-leaf checks, exact alpha validation against `CANDIDATE_ALPHAS`, and leaf computation `source + alpha * (proposal - source)`. Preserve alpha-one proposal leaves bit-exactly.

- [ ] **Step 4: Write Gaussian proximity RED tests**

```python
def test_gaussian_proximity_matches_closed_form_and_zero_identity():
    old = np.zeros((2, 2))
    new = np.array([[1.0, 0.0], [0.0, 2.0]])
    std = np.array([2.0, 4.0])
    epsilon = np.array([[0.5, -0.5], [1.0, 0.0]])
    report = gaussian_proximity(old, new, std, epsilon)
    np.testing.assert_allclose(report["kl_by_state_action"], [[0.125, 0], [0, 0.125]])
    assert gaussian_proximity(old, old, std, epsilon)["mean_kl"] == 0.0
```

Also test nonpositive/wrong-shape standard deviations, nonfinite inputs, and a tampered summary.

- [ ] **Step 5: Implement Gaussian KL and empirical log ratio**

Use the common old-policy sample `action = old_mean + std * epsilon`. Compute per-action KL, per-state summed log ratio, mean KL, per-state p95/max KL, log-ratio mean/p95-absolute/max-absolute, and a `kl_feasible` boolean from the fixed thresholds. Return raw arrays for independent artifact validation.

- [ ] **Step 6: Write selector RED tests**

Cover largest-alpha selection, a KL-feasible phase regression, preservation without advancement, no nonzero KL candidate, outcome precedence, and rejection of duplicate/missing alphas or altered source floors.

- [ ] **Step 7: Implement fail-closed selector**

Validate exactly six ordered nonzero rows. Eligibility is `kl_feasible and all(candidate >= floor)`. Select maximum alpha, then classify in this exact order: invalid execution, advances, preserves only, KL-feasible but unsafe, no nonzero KL candidate.

- [ ] **Step 8: Verify and commit Task 1**

Run focused pytest, Ruff, py_compile, and `git diff --check`, then commit:

```bash
git add tools/evaluate_g1_e023_kl_behavioral_trust.py tests/test_g1_e023_kl_behavioral_trust.py
git commit -m "feat: add KL behavioral trust contracts"
```

### Task 2: Exact one-update E023 proposal runner

**Files:**
- Create: `tools/run_g1_e023_kl_trust_proposal.py`
- Create: `tests/test_g1_e023_kl_trust_proposal.py`

**Interfaces:**
- Consumes: `build_rmr_noise_h24_kwargs`, E023 checkpoint/hparams hashes, existing training artifact validator.
- Produces: `build_e023_kl_trust_proposal_kwargs(...) -> dict[str, Any]`
- Produces: `validate_preflight(...) -> dict[str, Any]`
- Produces: exactly one checkpoint at transition `1_585_152`.

- [ ] **Step 1: Write exact-delta runner RED tests**

Assert the runner changes only `resume_from`, `total_steps`, and `checkpoint_interval` relative to E023; keeps exponential velocity reward, action-noise schedule endpoint `1_572_864`, effective-512 H24, zero bootstrap, CAGrad, no reset noise, and no domain randomization.

- [ ] **Step 2: Observe runner RED**

Run the focused module and require missing production module failure.

- [ ] **Step 3: Implement runner and provenance gate**

Reuse E023's reviewed builder and source hashes. Require clean exact 40-character HEAD, E023 checkpoint/hparams/reference/model/controller hashes, seed zero, g1-4x5, start step `1_572_864`, end step `1_585_152`, and one checkpoint. Persist atomic `preflight.json` and `training_validation.json`.

- [ ] **Step 4: Add artifact-validation tests**

Require loadable finite TrainState, exact step, exact checkpoint set, complete finite five-bin CAGrad telemetry, finite positive actor gradient/update norms, zero bootstrap, and the exact fixed final E023 noise vector. Reject corrupt pickle, extra checkpoint, absent telemetry, NaN, or hparam drift.

- [ ] **Step 5: Verify and commit Task 2**

Run focused tests and neighboring continuation-runner tests, Ruff, py_compile, and diff check, then commit only the runner and its tests.

### Task 3: Candidate publication and exact evaluation

**Files:**
- Modify: `tools/evaluate_g1_e023_kl_behavioral_trust.py`
- Modify: `tests/test_g1_e023_kl_behavioral_trust.py`

**Interfaces:**
- Consumes: source/proposal TrainStates, E023 audit bank, exact action-noise vector, existing Flax actor and phase-grid evaluator.
- Produces: `candidate_manifest.json`, `proximity_evidence.npz`, `proximity_summary.json`, seven phase-grid JSON files, `selection.json`, and selected rollout media when eligible.

- [ ] **Step 1: Write checkpoint-builder RED tests**

Use small actor and normalizer fixtures. Require each output state to contain only interpolated actor parameters and the source normalizer bit-exactly, carry its alpha in a hash-bound manifest, and be labeled `evaluation_only: true` and `resumable: false`.

- [ ] **Step 2: Implement atomic candidate publication**

Load source/proposal once, validate identical actor structure and source identity, create six compact `EvaluationActorState(actor_params, source_normalizer)` records, write each pickle through a same-directory temporary plus `os.replace`, then hash every checkpoint into the atomic manifest.

- [ ] **Step 3: Write real-bank actor/KL RED tests**

On four frozen audit-bank histories, verify source actions are finite 29-vectors, candidate actions use the source normalizer, identity KL is exactly zero, epsilon is taken from `noise_tape_a[:, 0, :]`, and the published NPZ independently recomputes the JSON summary.

- [ ] **Step 4: Implement actor and proximity evaluation**

Instantiate the exact E023 `Actor(hidden=(512,256,128), layer_norm=True, zero_output=False, squash=False)`. Normalize flattened 10x328 histories with the source `Normalizer`, evaluate source/candidates in float32, use `RMR_WALK_MODEL_999_ACTION_STD`, publish raw means/epsilon/KL/log-ratios and a validated summary.

- [ ] **Step 5: Write phase-grid orchestration and selector RED tests**

Require exact phases, compiled boundary metadata, checkpoint/reference/code hashes, all raw condition rows, source floor identity, and recomputation of the selector from phase-grid JSON. Reject stale summaries, wrong solver, wrong kernel, missing candidate, or a video whose checkpoint hash differs from the selection.

- [ ] **Step 6: Implement deterministic evaluation orchestration**

Call the existing Flax phase-grid evaluator for source and six candidates with `g1-4x5`, phases `0 25 50 75 100`, seed zero, and the checkpoint-restored exponential kernel. Aggregate only after every subprocess succeeds and every artifact revalidates. Render only a selected nonzero candidate; otherwise publish no misleading treatment video.

- [ ] **Step 7: CPU smoke and verification**

Run a no-render four-state real-checkpoint smoke with `JAX_PLATFORMS=cpu`, focused tests, the affected phase-grid suite, Ruff, py_compile, and diff check. Commit the evaluator repair.

### Task 4: Register, execute, and curate the bounded experiment

**Files:**
- Create in lab repo: `research/experiments/E-20260821-000.yaml`
- Modify after evidence only: `research/state/current.yaml`
- Generate after evidence only: vault Dashboard/experiment projection and local preview manifest.

**Interfaces:**
- Consumes: clean pinned code commit from Tasks 1-3 and exact E023/E016 assets.
- Produces: one preregistered outcome and canonical evidence under `runs/E-20260821-000/`.

- [ ] **Step 1: Write the schema-v1 experiment record**

Set `parent_id: E-20260814-023`, thread `g1-tracking-shac`, problem tags `closed-loop-transfer` and `optimizer-instability`, method tags `mjx`, `shac`, and `cagrad`. Copy the design's causal change, five-outcome map, exact artifacts, hashes, seed, one-GPU budget, and command.

- [ ] **Step 2: Validate and dry-run**

Run:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260821-000 --dry-run
```

Require both to exit zero before approval.

- [ ] **Step 3: Execute one guarded run**

Approve in-scope simulation as Codex, then run exactly:

```bash
uv run python -m tools.runexp E-20260821-000
```

Do not queue another experiment. Monitor compilation, the one update, candidate publication, KL evidence, and phase-grid evaluation.

- [ ] **Step 4: Curate only deterministic evidence**

Recompute hashes and classification from raw artifacts, select exactly one preregistered outcome, update the experiment and current state, and publish video/contact sheet only for an eligible selected candidate.

- [ ] **Step 5: Final verification and scoped publication**

Validate the registry, rebuild the vault, run focused registry tests, verify preview files exist, commit only owned lab files, and push the code branch plus lab commit. Report the measured KL, each candidate survival vector, selected outcome, and the exact next decision.
