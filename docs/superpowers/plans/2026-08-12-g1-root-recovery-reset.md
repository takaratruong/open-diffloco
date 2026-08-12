# G1 Root-Recovery Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off, resume-authorized root-focused reset mixture and run one exact E017-matched treatment from E012 final.

**Architecture:** The G1 environment samples one recovery-cohort Bernoulli per reset and applies a scalar multiplier only to root pose/velocity limits. SHAC owns validation, checkpoint restoration authority, environment transport, and hparams evidence. An experiment-specific runner inherits E012 exactly, changes only endpoint plus the two treatment settings, and validates the resulting dense checkpoint grid.

**Tech Stack:** Python 3.11, JAX/MJX, existing SHAC trainer, pytest, Ruff, DiffSim2Real registry.

## Global Constraints

- Defaults preserve the existing reset path exactly.
- Treatment is probability `0.5`, multiplier `2.0`, joint noise unchanged.
- Resume exact E012 final step `1720320`; end at `2113536`.
- Assistance remains exact zero for every treatment update.
- No treatment value or cohort flag enters actor observations.

---

### Task 1: Root-focused environment mixture

**Files:**
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `tests/test_g1_tracking_environment.py`

**Interfaces:**
- Consumes: `reference_reset_noise_scale: float` and the reset PRNG.
- Produces: constructor settings `reference_root_reset_noise_multiplier: float = 1.0` and `reference_root_reset_noise_probability: float = 0.0`; `_noisy_reference_state(..., root_multiplier: jax.Array | float = 1.0)`.

- [ ] **Step 1: Write failing constructor, default-equivalence, bound, and unchanged-joint tests**

Use the same key for legacy/default environments and require exact qpos/qvel equality. For treatment resets, reconstruct quaternion rotation error and require root translation/velocity inside 2× bounds while joints remain inside `0.050001` radians.

- [ ] **Step 2: Verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_tracking_environment.py -k 'root_reset_noise or reference_reset_noise'`

Expected: failures because the constructor settings do not exist.

- [ ] **Step 3: Implement validation and cohort sampling**

Validate finite multiplier `>=1.0` and finite probability in `[0,1]`. Split one extra `root_cohort_key` only when probability is positive, sample Bernoulli, and pass `where(recovery, multiplier, 1.0)` into `_noisy_reference_state`. Multiply only `pose_limit` and `velocity_limit`; leave `joint_delta` on the base scale.

- [ ] **Step 4: Verify GREEN and legacy neighbors**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_tracking_environment.py -k 'reset_noise or adaptive_resets'`

- [ ] **Step 5: Commit**

```bash
git add src/envs/g1_tracking/environment.py tests/test_g1_tracking_environment.py
git commit -m "feat: add root-focused reset mixture"
```

### Task 2: SHAC resume authority and evidence transport

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Create: `tests/test_shac_root_recovery_reset.py`

**Interfaces:**
- Produces: `resolve_root_reset_noise_resume_settings(resumed_hparams, *, requested_multiplier, requested_probability, allow_change) -> tuple[float, float]`.
- Extends: `train(..., reference_root_reset_noise_multiplier=1.0, reference_root_reset_noise_probability=0.0, allow_resume_reference_root_reset_noise_change=False)`.

- [ ] **Step 1: Write failing pure resolver and train-signature tests**

Require legacy checkpoints to resolve to `(1.0,0.0)`, default requests to restore checkpoint settings, mismatches to raise without authority, and explicit authority to admit `(2.0,0.5)`.

- [ ] **Step 2: Verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_root_recovery_reset.py`

- [ ] **Step 3: Implement validation, resume resolution, environment transport, and hparams persistence**

Call the resolver immediately after the carried-reset resolver. Pass both values into `g1_environment_kwargs` and record both plus the authority flag in `hparams.json`.

- [ ] **Step 4: Verify GREEN and resume neighbors**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_root_recovery_reset.py tests/test_shac_resume_randomness.py tests/test_shac_cagrad_integration.py`

- [ ] **Step 5: Commit**

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_root_recovery_reset.py
git commit -m "feat: plumb root-recovery reset treatment"
```

### Task 3: Exact treatment runner and experiment registration

**Files:**
- Create: `tools/run_g1_root_recovery_continuation.py`
- Create: `tests/test_g1_root_recovery_continuation_runner.py`
- Create after code commit: `research/experiments/E-20260812-002.yaml` in the lab registry.

**Interfaces:**
- Produces: `build_root_recovery_continuation_kwargs(...) -> dict`, `expected_checkpoint_steps()`, immutable preflight, and post-training validation.
- Consumes: exact E012 final checkpoint and E017's endpoint/cadence.

- [ ] **Step 1: Write failing exact-delta and validation tests**

Assert the only kwargs changed relative to E012 are `total_steps=2113536`, multiplier `2.0`, probability `0.5`, and the explicit resume-authority flag. Require the same eight checkpoint steps as E017 and exact-zero assistance telemetry.

- [ ] **Step 2: Verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_root_recovery_continuation_runner.py`

- [ ] **Step 3: Implement the minimal inherited runner**

Reuse E017's provenance and validation boundary. Add treatment hparams to postvalidation and keep parser free of scientific overrides.

- [ ] **Step 4: Verify the full affected slice**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_root_recovery_continuation_runner.py tests/test_g1_zero_assistance_consolidation_runner.py tests/test_shac_root_recovery_reset.py tests/test_g1_tracking_environment.py -k 'root_recovery or root_reset_noise or zero_assistance or reference_reset_noise'`

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check src/envs/g1_tracking/environment.py src/algorithms/shac/algorithm.py tools/run_g1_root_recovery_continuation.py tests/test_shac_root_recovery_reset.py tests/test_g1_root_recovery_continuation_runner.py`

- [ ] **Step 5: Commit and register**

Commit code, pin its full SHA in E018, validate the registry, dry-run, approve, and execute one guarded GPU experiment. Evaluate all eight checkpoints in two four-GPU waves and curate exactly one preregistered outcome.
