# G1 MJX PPO Positive Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a from-scratch PPO policy in the exact MJX G1 tracking environment used by DiffSim.

**Architecture:** Add an isolated PPO learner that consumes the existing `EnvState`, RMR actor/critic networks, and environment without changing task code. Add one guarded runner and reuse the deterministic five-phase evaluator for scientific selection.

**Tech Stack:** Python 3.11, JAX, Flax, Optax, MuJoCo MJX, pytest.

## Global Constraints

- Keep `g1_tracking_rmr_50hz_action_parity`, solver `g1-4x5`, reference-delta full-scale source-order actions, nominal physics, exact uniform reference-state reset, existing rewards, and existing termination unchanged.
- Disable observation noise, reset perturbation, domain randomization, pushes, assistance, and simulator gradients.
- Initialize learned diagonal action standard deviation to exactly `0.2`.
- Stop rollout data before every PPO loss; no gradient may traverse `env.step`.
- Do not modify SHAC behavior or checkpoint compatibility.
- Require deterministic five-phase trajectory, metrics, MP4, and montage before a competence claim.

---

### Task 1: PPO math and state

**Files:**
- Create: `src/algorithms/ppo/__init__.py`
- Create: `src/algorithms/ppo/algorithm.py`
- Test: `tests/test_g1_mjx_ppo.py`

**Interfaces:**
- Consumes: `GaussianRmrActorParams`, `RmrMlpParams`, `apply_rmr_mlp`, and `NormState`.
- Produces: `PPOTrainState`, `gaussian_log_prob`, `compute_gae`, and `ppo_loss`.

- [ ] Write tests proving Gaussian log probabilities, true-terminal versus truncation GAE, PPO ratio clipping, finite gradients, and initial standard deviation `0.2`.
- [ ] Run `pytest -q tests/test_g1_mjx_ppo.py` and observe missing-module RED.
- [ ] Implement only the pure PPO math and immutable train state required by those tests.
- [ ] Run `pytest -q tests/test_g1_mjx_ppo.py` and require GREEN.
- [ ] Commit `src/algorithms/ppo/` and `tests/test_g1_mjx_ppo.py`.

### Task 2: Exact-environment rollout and update

**Files:**
- Modify: `src/algorithms/ppo/algorithm.py`
- Modify: `tests/test_g1_mjx_ppo.py`

**Interfaces:**
- Consumes: `G1TrackingRMR50HzActionParityEnv.reset`, `.step`, actor observations, `bootstrap_critic_obs`, and `terminal`.
- Produces: `collect_rollout`, `update_ppo`, and `train`.

- [ ] Add tests proving the collector uses the exact environment action unchanged, stops all rollout leaves, preserves true terminal separately from clip-end done, and performs one finite parameter-changing PPO update.
- [ ] Run the focused tests and observe the new assertions fail.
- [ ] Implement vmapped reset/step collection, observation normalization, GAE, shuffled minibatch PPO epochs, Adam updates, finite checks, and atomic checkpoint/telemetry writes.
- [ ] Run the focused tests and require GREEN.
- [ ] Commit the rollout and trainer changes.

### Task 3: Guarded positive-control runner

**Files:**
- Create: `tools/run_g1_mjx_ppo.py`
- Create: `tests/test_g1_mjx_ppo_runner.py`

**Interfaces:**
- Consumes: `src.algorithms.ppo.algorithm.train`, solver profiles, and the exact model/controller/reference files.
- Produces: an output directory with checkpoints, `hparams.json`, and `training_metrics.jsonl`.

- [ ] Add tests that pin the exact environment, solver, reference, nominal distribution, initial standard deviation, 4,096 environments, H24 rollout, and reject dirty or mismatched provenance.
- [ ] Run the runner tests and observe missing-runner RED.
- [ ] Implement the runner with a 32-iteration early gate and a 512-iteration maximum registered budget.
- [ ] Run runner plus PPO tests and require GREEN; run Ruff, `py_compile`, and `git diff --check`.
- [ ] Commit the runner and tests.

### Task 4: Register, execute, and curate the PPO control

**Files:**
- Create in lab repository: `research/experiments/E-20260813-021.yaml`
- Modify after evaluation: `research/state/current.yaml`
- Modify generated projection: `docs/vault/Dashboard.md` and `docs/vault/experiments/E-20260813-021.md`

**Interfaces:**
- Consumes: guarded runner output and existing deterministic five-phase evaluator.
- Produces: a curated PPO-positive-control result with canonical video and montage.

- [ ] Register one causal question, outcome map, hashes, seed, GPU count, 32-iteration gate, 512-iteration ceiling, command, and evidence requirements.
- [ ] Validate and dry-run with `uv run python tools/researchctl.py validate` and `uv run python -m tools.runexp E-20260813-021 --dry-run`.
- [ ] Execute the guarded experiment; stop at the early gate unless finite learning and survival justify extension.
- [ ] Evaluate eligible checkpoints on phases `0,25,50,75,100`; require trajectory, JSON, MP4, and montage.
- [ ] Curate only a preregistered result, rebuild the vault, run focused registry tests, commit owned files, and publish the selected preview to the Dashboard cache.
