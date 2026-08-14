# Fresh PPO-Action-Contract Walking SHAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one provenance-bound fresh SHAC experiment that differs from E019 only by matching the competent PPO action contract.

**Architecture:** Add a narrow experiment runner that derives the established E019 kwargs and replaces only the environment action interface and residual scale. Reuse the existing trainer, evaluator, selection, and dashboard machinery; add fail-closed contract tests before launch.

**Tech Stack:** Python, JAX/MJX, Flax SHAC, pytest, Ruff, plain-file DiffSim2Real registry.

## Global Constraints

- Load no trained policy, critic, optimizer, normalizer, or PRNG state.
- Preserve the E019 recipe exactly except `env_variant`, `reference_residual_scale`, `squash_actor_mean`, and `clip_sampled_actor_actions` implied by that variant.
- Run one seed, one fixed 128-update budget, and no sweep.
- Require replay-free five-phase evaluation plus video, montage, and action diagnostics.

---

### Task 1: Bind the fresh PPO action contract

**Files:**
- Create: `tools/run_g1_fresh_ppo_action_contract_walk.py`
- Create: `tests/test_g1_fresh_ppo_action_contract_walk.py`

**Interfaces:**
- Consumes: `build_fresh_fixed_noise_kwargs(profile_name, reference_path, seed, actor_lr=1e-3, actor_per_env_grad_clip=1.0)`.
- Produces: `build_fresh_ppo_action_contract_kwargs(profile_name: str, reference_path: str | Path, seed: int) -> dict[str, Any]`, preflight validation, training artifact validation, and a CLI compatible with `tools.runexp`.

- [ ] **Step 1: Write failing contract tests**

Assert that the new builder is identical to E019 except for
`env_variant="g1_tracking_rmr_50hz_action_parity"` and
`reference_residual_scale=1.0`; instantiate the environment and assert
`squash_actor_mean is False` and `clip_sampled_actor_actions is False`. Assert
fresh initialization, fixed noise 0.2, LR 1e-3, zero bootstrap, H12/effective
512, CAGrad, and clip 1.0.

- [ ] **Step 2: Verify RED**

Run:
`/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_g1_fresh_ppo_action_contract_walk.py -q`

Expected: collection fails because the new runner module does not exist.

- [ ] **Step 3: Implement the minimal runner**

Derive the immutable kwargs from E019's builder, change only the two explicit
kwargs, reuse asset/code/reference preflight, call `train` under the pinned
solver context, and validate the persisted hparams and checkpoint telemetry.

- [ ] **Step 4: Verify GREEN and neighboring contracts**

Run:
`/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_g1_fresh_ppo_action_contract_walk.py tests/test_g1_fresh_fixed_noise_training.py tests/test_g1_rmr_action_space_parity_runner.py -q`

Expected: all tests pass.

- [ ] **Step 5: Run static checks and commit**

Run Ruff on the two new files, `python -m py_compile` on the runner, and
`git diff --check`. Commit only those files.

### Task 2: Register and execute the experiment

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260814-021.yaml`

**Interfaces:**
- Consumes: the clean implementation commit from Task 1.
- Produces: one immutable run under `runs/E-20260814-021/`.

- [ ] **Step 1: Preregister the causal comparison**

Pin the implementation commit, clean patch hash, environment, model,
controller, walking reference, g1-4x5 solver, seed zero, GPU count one, budget,
outcome map, exact command, and required evidence from the design.

- [ ] **Step 2: Validate and dry-run**

Run `uv run python tools/researchctl.py validate` and
`uv run python -m tools.runexp E-20260814-021 --dry-run`.

- [ ] **Step 3: Launch the guarded run**

Run `uv run python -m tools.runexp E-20260814-021`. Inspect the first saved
checkpoint for finite gradients, all five occupied CAGrad bins, post-clip norms
at most 1.0, and finite action diagnostics. Stop only for an invalid execution
gate, not because an early rollout is weak.

### Task 3: Evaluate and curate

**Files:**
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260814-021.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generate: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/Dashboard.md`
- Generate: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/experiments/E-20260814-021.md`

**Interfaces:**
- Consumes: all four checkpoint archives and validated telemetry.
- Produces: deterministic five-phase selection, canonical rollout artifacts, dashboard media, and one curated decision.

- [ ] **Step 1: Evaluate all checkpoints**

Run clean replay-free phases 0/25/50/75/100 for updates 16/32/64/128 and
record action means, sampled/effective actions, physical target offsets,
survival, reward, and tracking errors.

- [ ] **Step 2: Apply the registered gate**

Select only among finite valid checkpoints. Classify as solve, advance,
insufficient, or invalid using the exact design thresholds; do not tune after
seeing the result.

- [ ] **Step 3: Publish evidence and curate**

Copy the selected MP4/montage/diagnostic plot to the untracked vault preview
path, update the experiment result and current state, run registry validation,
rebuild the vault, and commit only owned canonical records.
