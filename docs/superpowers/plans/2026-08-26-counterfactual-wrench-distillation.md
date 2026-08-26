# Counterfactual Wrench Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an exact-zero-wrench, strictly leg-only residual to reproduce the local closed-loop effect of the successful frozen E004 torso-wrench teacher.

**Architecture:** Add a focused module for the canonical twelve-leg action scatter, counterfactual transition target, normalized loss, and feasibility projection. A guarded evaluator first tests whether the teacher effect is locally reachable through the legs. Only after that gate passes, extend the existing frozen-controller residual path with a 12-output adapter and a counterfactual actor-objective term while retaining five-phase CAGrad, then run one fixed 32-update continuation and deterministic CPU selection.

**Tech Stack:** Python 3.11, JAX, MJX/MuJoCo, Flax, Optax, NumPy, pytest, Ruff, the existing SHAC trainer and research registry.

## Global Constraints

- Freeze the complete E026 controller, normalizer, optimizer snapshot, and E004 wrench head.
- The student applies bit-exact zero `xfrc_applied` at the torso during training and evaluation.
- Train only the twelve canonical hip, knee, and ankle residual outputs; waist and upper-body corrections remain bit-exact zero.
- Use exact E004 update 16, the continuous reference, solver `g1-4x5-fixed-scan`, seed zero, effective-512 H24, and deterministic CPU selection.
- Fixed objective constants: teacher-RMS normalization floor `1e-3`, pseudo-Huber delta `0.1`, four block weights `0.25`, residual magnitude weight `0.01`, temporal-change weight `0.001`.
- No sweep, assistance schedule, replay, pushes, observation noise, physics randomization, phase reward, named-foot reward, or capture reward.
- The feasibility gate requires all five phases and median normalized projection residual `<= 0.50`; a failed gate prevents training.
- A retained checkpoint must componentwise preserve E026 `131/114/74/71/74` and strictly improve at least one phase.

---

### Task 1: Canonical Leg Residual And Counterfactual Loss

**Files:**
- Create: `src/algorithms/shac/counterfactual_wrench_distillation.py`
- Modify: `src/algorithms/shac/frozen_controller_residual.py`
- Test: `tests/test_shac_counterfactual_wrench_distillation.py`

**Interfaces:**
- Consumes: canonical `actor_joint_names`, `PreviewResidualAdapter`, and `FrozenControllerResidualParams`.
- Produces: `LEG_ACTION_NAMES`, `resolve_leg_action_indices(actor_joint_names)`, `scatter_leg_residual(residual, indices, action_dim=29)`, `counterfactual_target_change(teacher_before, teacher_after)`, `counterfactual_transition_loss(student_change, teacher_change, target_rms)`, and optional `residual_action_indices` arguments on frozen-controller apply/migration.

- [ ] **Step 1: Write failing unit tests for exact joint order and scattering**

```python
def test_leg_residual_scatter_is_zero_outside_canonical_twelve():
    indices = resolve_leg_action_indices(RMR_ACTION_STD_JOINT_NAMES)
    scattered = scatter_leg_residual(jp.arange(12.0), indices, action_dim=29)
    assert scattered.shape == (29,)
    assert np.array_equal(np.asarray(scattered)[list(indices)], np.arange(12.0))
    assert np.count_nonzero(np.delete(np.asarray(scattered), indices)) == 0
```

Also reject missing, duplicate, reordered, nonfinite, or non-12 residual inputs.

- [ ] **Step 2: Run the focused RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_counterfactual_wrench_distillation.py`

Expected: collection fails because `src.algorithms.shac.counterfactual_wrench_distillation` does not exist.

- [ ] **Step 3: Implement the fixed leg interface and normalized objective**

```python
LEG_ACTION_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
)

def scatter_leg_residual(values, indices, *, action_dim=29):
    values = jp.asarray(values)
    if values.shape[-1] != 12 or len(indices) != 12:
        raise ValueError("leg residual requires exactly twelve actions")
    return jp.zeros(values.shape[:-1] + (action_dim,), values.dtype).at[..., jp.asarray(indices)].set(values)
```

Represent each counterfactual change as a `(12,)` concatenation of base linear/angular velocity and centroidal linear/angular momentum changes. The loss validates finite `(12,)` vectors and target RMS, applies RMS floor `1e-3`, computes pseudo-Huber delta `0.1` independently for four 3-vectors, and returns their equal-weight mean plus named telemetry.

- [ ] **Step 4: Extend frozen-controller residual application without changing the 29-output legacy path**

Add `residual_action_indices: tuple[int, ...] | None = None` to `apply_frozen_controller_residual` and `migrate_frozen_controller_residual`. When indices are absent, retain exact legacy shape checks. When present, require a 12-output adapter and scatter it before adding to the 29-D parent action. Return the scattered 29-D residual for telemetry.

- [ ] **Step 5: Run focused and legacy tests**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_counterfactual_wrench_distillation.py tests/test_shac_frozen_controller_residual.py`

Expected: all tests pass and legacy 29-D action equality remains exact.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/algorithms/shac/counterfactual_wrench_distillation.py src/algorithms/shac/frozen_controller_residual.py tests/test_shac_counterfactual_wrench_distillation.py
git commit -m "feat: add leg-only counterfactual wrench objective"
```

### Task 2: Frozen-Teacher Feasibility Discriminator

**Files:**
- Create: `tools/evaluate_g1_counterfactual_wrench_feasibility.py`
- Create: `tests/test_g1_counterfactual_wrench_feasibility.py`

**Interfaces:**
- Consumes: exact E004 checkpoint/hparams, E026 subtree, continuous reference, runtime model/controller, five phases, and Task 1 change/scatter helpers.
- Produces: `bounded_damped_projection(jacobian, target, lower, upper, damping)`, `classify_feasibility(rows, threshold=0.50)`, and atomic `counterfactual_wrench_feasibility.json/.npz` artifacts.

- [ ] **Step 1: Write failing pure contract tests**

```python
def test_projection_recovers_reachable_target():
    jacobian = np.eye(12)
    report = bounded_damped_projection(jacobian, np.ones(12) * 0.1,
                                        np.full(12, -1.0), np.full(12, 1.0), 1e-4)
    assert report["normalized_residual"] < 1e-3
    assert report["rank"] == 12

def test_gate_requires_every_phase_and_median_at_most_half():
    rows = fixture_rows(phases=(0, 25, 50, 75, 100), normalized_residual=0.49)
    assert classify_feasibility(rows)["outcome"] == "leg-counterfactual-feasible"
```

Tests also reject missing phases, zero targets, nonfinite Jacobians, malformed action bounds, wrong hashes, wrong teacher checkpoint, stale external force, and partial atomic artifacts.

- [ ] **Step 2: Run the focused RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_counterfactual_wrench_feasibility.py`

Expected: import failure for the missing evaluator.

- [ ] **Step 3: Implement exact preflight and projection helpers**

Load the E004 composite with the existing learned-wrench and nested-controller types. Hash the checkpoint, sibling hparams, E026 subtree, code commit/dirty patch, XML, controller, reference, and solver profile. Build action bounds from the exact E026 action and the configured action boundary. Use SVD-based damped least squares, clip once to bounds, recompute the achieved target, and store rank/singular values/residuals rather than treating the unconstrained solve as evidence.

- [ ] **Step 4: Implement matched MJX sampling**

For phases `0/25/50/75/100`, restore exact reference states, keep only stance rows with finite nonzero teacher wrench, compute the stop-gradient teacher one-step target, and use `jax.jacfwd` for the 12-leg residual-to-target Jacobian through the exact solver. Require exact zero torso force in the student probe. Write raw arrays to NPZ first, bind its SHA into JSON, then atomically replace the final JSON.

- [ ] **Step 5: Run pure tests and a one-state CPU smoke**

Run the focused tests, then the complete smoke command:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_counterfactual_wrench_feasibility.py
JAX_PLATFORMS=cpu /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m tools.evaluate_g1_counterfactual_wrench_feasibility \
  --teacher-checkpoint /home/ubuntu/projects/diffsim2real-lab/runs/E-20260825-004/20260825T162240Z/seed-0/training/training_runs/shac_20260825_092244/checkpoint_step_1966080.pkl \
  --reference-path /home/ubuntu/projects/diffsim2real/outputs/lafan_walk_win137_300_prefix_exact_named.npz \
  --model-path /home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml \
  --controller-path /home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz \
  --solver-profile g1-4x5 \
  --phases 0 25 50 75 100 \
  --max-states-per-phase 1 \
  --code-commit "$(git rev-parse HEAD)" \
  --output-dir /tmp/e008-feasibility-smoke
```

Expected: unit tests pass; smoke JSON is valid, hash-bound, covers the requested probe, and contains finite nonzero Jacobian evidence.

- [ ] **Step 6: Commit Task 2**

```bash
git add tools/evaluate_g1_counterfactual_wrench_feasibility.py tests/test_g1_counterfactual_wrench_feasibility.py
git commit -m "feat: add wrench-to-leg feasibility discriminator"
```

### Task 3: SHAC Counterfactual Teacher Integration

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `src/algorithms/shac/frozen_controller_residual.py`
- Create: `tests/test_shac_counterfactual_wrench_integration.py`

**Interfaces:**
- Consumes: Task 1 leg residual/objective, exact E004 teacher source path/SHA, fixed target RMS from Task 2, and the existing frozen-controller CAGrad path.
- Produces: fail-closed `train()` flags, exact resume metadata, per-update counterfactual telemetry, and checkpoint-bound frozen/non-leg/zero-wrench audits. The counterfactual term augments the actor objective; existing CAGrad continues to combine five phase-bin gradients of that complete objective.

- [ ] **Step 1: Write failing train-interface and source-validation tests**

Require defaults disabled; enabled mode requires frozen-controller residual, exactly twelve registered indices, E004 teacher path/SHA, a 12-vector finite positive target RMS, zero torso assistance, no learned-wrench application on the student, and explicit resume authority. Assert the legacy trainer source and hparams remain unchanged when disabled.

- [ ] **Step 2: Run the focused RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_counterfactual_wrench_integration.py`

Expected: missing train parameters and integration seam assertions fail.

- [ ] **Step 3: Add the 12-output migration and immutable teacher source**

Initialize `PreviewResidualAdapter(action_dim=12, hidden_dim=256)` only in counterfactual mode. Load the exact E004 teacher once before JIT, validate its complete tree and source hashes, and capture it as a frozen JAX pytree. Build the student migration from exact E026 and confirm the zero-head action is bit-identical before the first update.

- [ ] **Step 4: Add the local teacher transition inside the rollout**

Immediately before the real student `env.step`, clone the current state, overwrite the teacher torso wrench, and step it with the frozen E026 action. Stop gradients through the teacher next state. Independently clear the student's torso `xfrc_applied`, step with E026 plus the scattered leg residual, compute both 12-D changes, and add the fixed counterfactual task plus residual magnitude/temporal penalties. Carry only the student state.

- [ ] **Step 5: Add masks, telemetry, and persistence**

Persist finite task loss, each four-block loss, normalized target/achieved RMS, cosine, correction RMS/max, bound activity, exact-zero student wrench, non-leg residual max, frozen teacher/parent/normalizer drift, and the feasibility artifact SHA. Fail the update if any validity field is false. Include all settings in hparams and resume comparison.

- [ ] **Step 6: Run focused and neighboring suites**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_shac_counterfactual_wrench_distillation.py \
  tests/test_shac_counterfactual_wrench_integration.py \
  tests/test_shac_frozen_controller_residual.py \
  tests/test_shac_learned_torso_wrench_integration.py \
  tests/test_shac_cagrad_integration.py
```

Expected: all tests pass; disabled legacy behavior remains exact.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/algorithms/shac/algorithm.py src/algorithms/shac/frozen_controller_residual.py tests/test_shac_counterfactual_wrench_integration.py
git commit -m "feat: train leg residual from frozen wrench counterfactuals"
```

### Task 4: Guarded Experiment Runner And Deterministic Selector

**Files:**
- Create: `tools/run_g1_counterfactual_wrench_distillation.py`
- Create: `tests/test_g1_counterfactual_wrench_runner.py`
- Modify: `tools/evaluate_g1_tracking.py`
- Modify: `tests/test_g1_tracking_evaluator.py`

**Interfaces:**
- Consumes: exact Task 2 feasibility artifact with passing outcome, Task 3 train flags, E004/E026 assets, and existing nested-policy evaluator.
- Produces: one guarded 32-update run, four checkpoints, deterministic CPU phase grids, `selection.json`, learning curves, MP4, and contact sheet.

- [ ] **Step 1: Write runner RED tests**

Assert an exact delta from the E007/E004 builders: 32 effective-512 H24 updates, checkpoints 8/16/24/32, counterfactual mode enabled, all forbidden treatments disabled, feasibility threshold fixed, and no CLI sweep. Reject a missing/failing/tampered feasibility artifact, wrong code/assets, extra checkpoints, nonfinite telemetry, nonzero wrench, non-leg action, or frozen-tree drift.

- [ ] **Step 2: Implement preflight, train, and artifact validation**

Use atomic reports and exact checkpoint-step enumeration. Validate every checkpoint by unpickling state, checking step/finiteness/tree hashes, and corroborating checkpoint telemetry. A process failure or invalid first update terminates without scientific classification.

- [ ] **Step 3: Extend evaluator only as needed for the 12-output nested adapter**

The evaluator must scatter the leg correction with the exact saved indices, require zero non-leg residuals and zero torso wrench, and persist raw parent/leg/scattered/effective actions. Disabled evaluator behavior remains exact.

- [ ] **Step 4: Implement CPU phase selection and visual artifacts**

Evaluate all four checkpoints at phases `0/25/50/75/100` on CPU. Select only candidates that componentwise meet `131/114/74/71/74` and strictly improve one phase; rank eligible candidates by minimum, median, mean, then earlier update. Render only the selected eligible policy; if none is eligible, render the descriptive best candidate while marking `policy_retained=false`.

- [ ] **Step 5: Run focused tests and static checks**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_counterfactual_wrench_runner.py tests/test_g1_tracking_evaluator.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check tools/run_g1_counterfactual_wrench_distillation.py tools/evaluate_g1_tracking.py tests/test_g1_counterfactual_wrench_runner.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile tools/run_g1_counterfactual_wrench_distillation.py
git diff --check
```

Expected: tests and static checks pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add tools/run_g1_counterfactual_wrench_distillation.py tools/evaluate_g1_tracking.py tests/test_g1_counterfactual_wrench_runner.py tests/test_g1_tracking_evaluator.py
git commit -m "feat: run and select counterfactual wrench distillation"
```

### Task 5: Register, Execute, Evaluate, And Curate

**Files:**
- Create in lab repository: `research/experiments/E-20260826-000.yaml`
- Modify after evidence: `research/state/current.yaml`
- Build after evidence: `docs/vault/Dashboard.md`, `docs/vault/experiments/E-20260826-000.md`
- Create only if promotion rule is met: `docs/vault/findings/F-20260826-001 Counterfactual Wrench Distillation.md`

**Interfaces:**
- Consumes: clean code commit, passing focused suites, exact E004/E026/reference/runtime hashes, and guarded runner.
- Produces: preregistered feasibility outcome; if feasible, one training result and deterministic visual evidence.

- [ ] **Step 1: Register one causal experiment**

The outcome map is:

- `leg-counterfactual-advances`: feasibility passes and an unassisted checkpoint preserves every E026 phase while improving at least one;
- `leg-counterfactual-feasible-but-insufficient`: feasibility passes but no trained checkpoint is eligible;
- `leg-counterfactual-not-feasible`: median projection residual exceeds `0.50` or any phase lacks coverage, so training is skipped;
- `invalid-execution`: any provenance, finite-state, frozen-tree, artifact, or determinism gate fails.

- [ ] **Step 2: Validate and dry-run**

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260826-000 --dry-run
```

Expected: registry valid and dry-run resolves exact pinned inputs without compute.

- [ ] **Step 3: Execute the feasibility discriminator**

Run: `uv run python -m tools.runexp E-20260826-000`

Expected first branch: publish a valid feasibility artifact. If the outcome is `leg-counterfactual-not-feasible`, the guarded runner exits successfully without launching training.

- [ ] **Step 4: If and only if feasible, execute bounded training and CPU evaluation**

Monitor compilation and the first update for finite positive leg gradients, exact-zero wrench, and zero frozen/non-leg drift. Let the immutable 32-update budget finish, then validate all four checkpoints and deterministic CPU phase grids.

- [ ] **Step 5: Curate and publish visuals**

Update the experiment and current state from `selection.json`, run registry validation, rebuild the vault, copy the MP4/plots/contact sheet into the local Dashboard preview path, and publish the same canonical artifacts to the local HTML dashboard.

- [ ] **Step 6: Commit owned research records**

```bash
git add research/experiments/E-20260826-000.yaml research/state/current.yaml docs/vault/Dashboard.md docs/vault/experiments/E-20260826-000.md
git commit -m "research: evaluate counterfactual wrench distillation"
```
