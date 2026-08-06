# G1 Actuator-Authority Shift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed G1 torque-authority scale and a same-process screen that selects the least severe nonterminal, materially discriminative shift.

**Architecture:** The environment owns validation and applies the scale exactly once to the controller effort limits. SHAC and evaluation transport and record the scalar. A focused comparison CLI evaluates one source actor across nominal and ordered shifted environments in one process, applies a pure selection predicate, and atomically publishes strict finite JSON.

**Tech Stack:** Python 3.11, unittest, NumPy, JAX/MJX, argparse, JSON.

## Global Constraints

- Preserve current behavior byte-for-byte at `effort_limit_scale=1.0`.
- The scale must be finite and strictly positive.
- Change no PD gain, action scale, observation, reward, termination, solver, reference, mass, or timing behavior.
- Screen scales in registered input order and never select a terminal shift.
- Publish JSON atomically with `allow_nan=False`.

---

### Task 1: Environment torque-authority seam

**Files:**
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `tests/test_g1_tracking_environment.py`

**Interfaces:**
- Consumes: `G1TrackingEnv(..., effort_limit_scale: float = 1.0)`
- Produces: `env.effort_limit_scale: float` and scaled `env.effort_limit`

- [ ] **Step 1: Write failing environment tests**

Add tests that construct scale `0.7`, assert the recorded scale, assert
`shifted.effort_limit == nominal.effort_limit * 0.7`, assert nominal limits
equal the controller tape, and reject `0.0`, `-1.0`, `nan`, and `inf`.

- [ ] **Step 2: Verify red**

Run:

```bash
python -m unittest tests.test_g1_tracking_environment.G1TrackingEnvironmentTest.test_fixed_effort_limit_scale_changes_only_torque_authority tests.test_g1_tracking_environment.G1TrackingEnvironmentTest.test_effort_limit_scale_rejects_nonpositive_or_nonfinite_values -v
```

Expected: failure because `effort_limit_scale` is ignored and no validation
exists.

- [ ] **Step 3: Implement the minimal environment seam**

Add the constructor argument and validate it with:

```python
if not math.isfinite(effort_limit_scale) or effort_limit_scale <= 0.0:
    raise ValueError("effort_limit_scale must be positive and finite")
self.effort_limit_scale = float(effort_limit_scale)
```

Set:

```python
self.effort_limit = jp.asarray(
    self.controller.effort_limit * self.effort_limit_scale
)
```

- [ ] **Step 4: Verify green**

Run the two focused tests and the complete
`tests.test_g1_tracking_environment` module. Expected: pass.

### Task 2: Training and evaluation transport

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tools/run_g1_tracking_rmr50_shac.py`
- Modify: `tools/evaluate_g1_tracking.py`
- Modify: `tests/test_g1_tracking_runner.py`
- Modify: `tests/test_g1_tracking_evaluator.py`

**Interfaces:**
- Consumes: `build_train_kwargs(..., effort_limit_scale=...)`,
  `make_evaluation_env(..., effort_limit_scale=...)`, and
  `--effort-limit-scale`
- Produces: exact environment transport plus `hparams.json` and
  `summary.json` provenance

- [ ] **Step 1: Write failing transport tests**

Add runner tests for scale `0.7` and invalid `0.0`; evaluator tests for factory
transport and parser transport. Assert `kwargs["effort_limit_scale"] == 0.7`
and `env.effort_limit_scale == 0.7`.

- [ ] **Step 2: Verify red**

Run:

```bash
python -m unittest tests.test_g1_tracking_runner tests.test_g1_tracking_evaluator -v
```

Expected: failures for missing arguments/keys.

- [ ] **Step 3: Implement minimal transport**

Add `effort_limit_scale` to the RMR runner, validate it, add it to the SHAC
training kwargs and CLI, and pass it from `main`. Add the scalar to the SHAC
signature and hparams; pass it to `Go2Env` only when
`env_variant.startswith("g1_tracking")` so Go2 constructors remain unchanged.
Add the argument to `make_evaluation_env`, evaluator CLI, environment
construction, and summary.

- [ ] **Step 4: Verify green**

Run the runner/evaluator modules plus
`tests.test_g1_tracking_controller`. Expected: pass.

### Task 3: Same-process authority selector

**Files:**
- Create: `tools/compare_g1_tracking_effort_limit_scales.py`
- Create: `tests/test_g1_tracking_effort_limit_scale_comparison.py`

**Interfaces:**
- Consumes: source checkpoint, ordered scales, phases, seed, horizon, solver
- Produces: strict finite JSON with nominal results, shifted candidates,
  deltas, gate flags, and selected scale

- [ ] **Step 1: Write failing pure selection tests**

Create synthetic candidate documents and test:

```python
candidate_passes(candidate, minimum_reward_drop=0.001)
select_earliest_scale(candidates, minimum_reward_drop=0.001)
```

Cover the exact threshold, only three worsened errors, an added terminal,
earliest input-order selection, and null selection. Add a parser test for
`--effort-limit-scales 1.0 0.8 0.7 0.6 0.5`.

- [ ] **Step 2: Verify red**

Run:

```bash
python -m unittest tests.test_g1_tracking_effort_limit_scale_comparison -v
```

Expected: import error because the comparison module does not exist.

- [ ] **Step 3: Implement the comparator**

Load the source policy once. Require a unique first scale of exactly `1.0`,
unique positive finite scales, valid phases, and positive
`--minimum-reward-drop` (default `0.001`). Evaluate nominal once, then each
shifted environment in input order using the existing `rollout`, `aggregate`,
and `summary_delta` functions. Define a pass as:

```python
added_terminal = shifted["terminal_count"] > nominal["terminal_count"]
reward_drop = nominal["mean_reward"] - shifted["mean_reward"]
worsened_errors = sum(delta[field] > 0.0 for field in TRACKING_ERROR_FIELDS)
return not added_terminal and reward_drop >= minimum_reward_drop and worsened_errors >= 4
```

Select the first passing shifted candidate. Recursively reject non-finite
floats, write to a dot-prefixed temporary sibling with `allow_nan=False`, then
replace the destination atomically.

- [ ] **Step 4: Verify green**

Run the new module and neighboring residual/full-policy comparator test
modules. Expected: pass.

### Task 4: Verification and publication

**Files:**
- Modify only if verification exposes an owned defect.

**Interfaces:**
- Consumes: all implementation and focused tests
- Produces: one reviewed, reproducible code commit suitable for an experiment
  card

- [ ] **Step 1: Run focused verification**

```bash
python -m unittest \
  tests.test_g1_tracking_environment \
  tests.test_g1_tracking_controller \
  tests.test_g1_tracking_runner \
  tests.test_g1_tracking_evaluator \
  tests.test_g1_tracking_effort_limit_scale_comparison \
  tests.test_g1_tracking_residual_comparison \
  tests.test_g1_tracking_full_policy_comparison \
  tests.test_g1_tracking_full_policy_checkpoint_comparison -v
python -m compileall -q src tools tests
git diff --check
```

- [ ] **Step 2: Review the exact diff**

Confirm scale `1.0` preserves the old path, non-G1 environments do not receive
the new keyword, JSON uses strict finite serialization, and no scientific
threshold is selected from observed screen results.

- [ ] **Step 3: Commit and push**

Stage only owned files, commit one implementation checkpoint, and push
`research/g1-rmr-50hz-20260805` to `takaratruong`.

- [ ] **Step 4: Register the GPU screen**

Create one approved experiment card pinned to the implementation commit for the
exact ordered scales `1.0/0.8/0.7/0.6/0.5`, phases `0/30/60/90`, seed zero,
60 steps, solver 4/5, and a 60-minute wall-time. Validate and dry-run before
launch.
