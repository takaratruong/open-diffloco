# Bounded-Mean Early Learning Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the decoupled G1 SHAC treatment for exactly 98,304 transitions, then fail closed unless its stochastic training rollout remains physically learnable and its clean rollout survives at least 40 steps.

**Architecture:** Extend the existing action-space parity runner with an explicit early-learning mode. Reuse the checkpoint-aware evaluator for the noisy and clean rollouts, but keep artifact validation and experiment advancement in the parent runner so no evaluator-side claim can pass without checkpoint, action-tape, solver, phase, and training telemetry agreement.

**Tech Stack:** Python, NumPy, JAX/MJX, MuJoCo, unittest/pytest, Ruff, the DiffSim2Real experiment registry.

---

## Task 1: Compile the exact 98k treatment

**Files:**

- Modify: `tools/run_g1_rmr_action_space_parity.py`
- Modify: `tests/test_g1_rmr_action_space_parity_runner.py`

- [ ] Add RED tests for `build_decoupled_early_learning_kwargs`: it must differ from the full decoupled treatment only in `total_steps=98_304`, `checkpoint_interval=98_304`, `curriculum_grace=98_304`, and `curriculum_steps=1`.
- [ ] Add RED parser tests proving `--early-learning-gate` is mutually exclusive with `--gate-only` and rejected unless `--decoupled-exploration` is present.
- [ ] Implement the builder and a small `validate_mode_args` boundary. Keep all actor, observation, reset, physics, CAGrad, noise schedule, solver, seed, and future-reference settings inherited from `build_decoupled_exploration_kwargs`.
- [ ] Route `execute` to the new builder without changing either legacy gate or full-run behavior.
- [ ] Run the focused parser/builder tests and commit the treatment compilation.

## Task 2: Render and validate the early-learning evidence

**Files:**

- Modify: `tools/run_g1_rmr_action_space_parity.py`
- Modify: `tests/test_g1_rmr_action_space_parity_runner.py`

- [ ] Add RED fixtures for an exact step-98,304 checkpoint, its hparams, five-bin CAGrad checkpoint row, positive actor gradient/update telemetry, a 120-step stochastic action tape, and a clean phase-zero summary.
- [ ] Add `render_decoupled_early_learning_rollouts`. Build the noisy command from the checkpoint with `--training-distribution-rollout --disable-training-observation-noise --exact-training-reset-phase 0 --max-steps 120 --seed 0`. Build the clean command from checkpoint hparams using the same environment, reference, history, future lookaheads, residual control, solver profile, and actor architecture.
- [ ] Add `validate_early_learning_artifacts`. Require the exact hparams contract; `checkpoint_step_098304.pkl`; a single step-98,304 telemetry row; five finite occupied CAGrad bins; positive finite actor gradient and update norm; checkpoint-bound noisy/clean summaries; and finite `(120, 29)` `action_mean`, `epsilon`, `noisy_action`, and `effective_action` arrays plus a finite `(29,)` standard deviation.
- [ ] Compute and publish mean RMS, mean max, mean saturation fraction at `abs(mean) >= 0.95`, noisy first-episode survival from the first terminal/done record, and clean phase-zero survival. Fail unless mean max is at most `1.000001`, mean saturation is below `0.20`, noisy evidence is complete, and clean survival is at least 40.
- [ ] Write `action_space_parity_early_learning_validation.json` atomically only after every gate passes. Add negative tests for saturation, malformed/nonfinite tapes, provenance mismatch, missing bins, and insufficient clean survival.
- [ ] Run focused tests, Ruff, `py_compile`, and `git diff --check`; self-review the scoped diff and commit it.

## Task 3: Review, register, execute, and curate E004

**Files:**

- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260813-004.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generate: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/experiments/E-20260813-004.md`
- Generate/cache: dashboard media under the vault experiment attachment/cache contract

- [ ] Request an independent read-only review of the runner and tests. Repair every Critical or Important launch blocker, repeat focused verification, and pin a clean 40-character code commit.
- [ ] Register E004 before compute with the exact code commit, reference/model/controller hashes, GPU 1, 98,304-transition budget, noisy and clean rollout commands, thresholds, and stop rule. Validate the registry and dry-run the registered command.
- [ ] Launch only through `uv run python -m tools.runexp E-20260813-004`. Monitor compilation and all 16 updates; stop on nonfinite telemetry or contract drift.
- [ ] Evaluate the final checkpoint through the runner. If the gate passes, preserve the noisy training-distribution episode and clean phase-zero rollout; if it fails, preserve the same evidence and record the failing threshold without silently extending training.
- [ ] Update the experiment result and current research state, rebuild/validate the vault, verify playable dashboard media, and commit the research curation. Open the noisy and clean videos for visual inspection.

## Verification commands

```bash
conda run -n diffsim pytest -q tests/test_g1_rmr_action_space_parity_runner.py
conda run -n diffsim ruff check tools/run_g1_rmr_action_space_parity.py tests/test_g1_rmr_action_space_parity_runner.py
conda run -n diffsim python -m py_compile tools/run_g1_rmr_action_space_parity.py tests/test_g1_rmr_action_space_parity_runner.py
git diff --check
```

In the lab repository:

```bash
uv run python -m tools.researchctl validate
uv run python -m tools.runexp E-20260813-004 --dry-run
uv run python -m tools.runexp E-20260813-004
uv run python tools/researchctl.py build-vault
uv run python -m tools.researchctl validate
```
