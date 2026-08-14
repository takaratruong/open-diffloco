# Fresh Full-Action H24 Walking SHAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one provenance-bound fresh SHAC experiment that changes E021 only from H12 to H24 while preserving 128 optimizer updates.

**Architecture:** Add a thin H24 runner derived from the tested E021 builder. Generalize only E021's artifact validator inputs so the H24 runner can reuse the same fail-closed checkpoint, CAGrad, action, and finite-state validation without copying it.

**Tech Stack:** Python, JAX/MJX, Flax SHAC, pytest, Ruff, plain-file DiffSim2Real registry.

## Global Constraints

- Change only `unroll_length`, `total_steps`, and `checkpoint_interval` relative to E021.
- Preserve exactly 128 optimizer updates and effective population 512.
- Use H24 total steps 1,572,864 and checkpoint interval 196,608.
- Load no PPO, BC, resume, critic, optimizer, normalizer, or PRNG state.
- Run seed zero only; no horizon, learning-rate, noise, or population sweep.
- Require exact five-phase evaluation and right-leg PPO comparison before interpretation.

---

### Task 1: Bind and validate the H24 treatment

**Files:**
- Create: `tools/run_g1_fresh_full_action_h24_walk.py`
- Create: `tests/test_g1_fresh_full_action_h24_walk.py`
- Modify: `tools/run_g1_fresh_ppo_action_contract_walk.py`
- Modify: `tests/test_g1_fresh_ppo_action_contract_walk.py`

**Interfaces:**
- Consumes: `build_fresh_ppo_action_contract_kwargs(...)` and `validate_training_artifacts(...)` from the E021 runner.
- Produces: `build_fresh_full_action_h24_kwargs(profile_name, reference_path, seed) -> dict[str, Any]`, `expected_checkpoint_steps() -> tuple[int, ...]`, H24 preflight validation, and a `tools.runexp`-compatible CLI.

- [ ] **Step 1: Write the failing H24 contract test**

```python
def test_h24_builder_changes_only_horizon_and_cadence() -> None:
    baseline = build_fresh_ppo_action_contract_kwargs(
        "g1-4x5", Path("/tmp/walk.npz"), 0
    )
    treatment = build_fresh_full_action_h24_kwargs(
        "g1-4x5", Path("/tmp/walk.npz"), 0
    )
    changed = {
        key for key in set(baseline) | set(treatment)
        if baseline.get(key) != treatment.get(key)
    }
    assert changed == {"unroll_length", "total_steps", "checkpoint_interval"}
    assert treatment["unroll_length"] == 24
    assert treatment["total_steps"] == 1_572_864
    assert treatment["checkpoint_interval"] == 196_608
    assert treatment["num_envs"] == 256
    assert treatment["gradient_accumulation_steps"] == 2
    assert treatment["total_steps"] // (256 * 2 * 24) == 128
    assert expected_checkpoint_steps() == tuple(
        range(196_608, 1_572_864 + 1, 196_608)
    )
```

- [ ] **Step 2: Verify RED**

Run:
`PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/pytest -q tests/test_g1_fresh_full_action_h24_walk.py`

Expected: collection fails because `tools.run_g1_fresh_full_action_h24_walk` does not exist.

- [ ] **Step 3: Generalize the existing validator without changing defaults**

Change the E021 validator signature to:

```python
def validate_training_artifacts(
    run_directory: Path,
    *,
    expected_kwargs: dict[str, Any] | None = None,
    expected_steps: tuple[int, ...] | None = None,
    total_steps: int = TOTAL_STEPS,
    protocol: str = "g1-fresh-ppo-action-contract-walk-training-v1",
) -> dict[str, Any]:
```

When the optional arguments are absent, construct the same E021 expected kwargs
and checkpoint steps as before. Use `total_steps` for final/latest checkpoint
validation and return the passed `protocol`. Add an E021 regression assertion
that calling the validator without overrides still requires E021's original
steps and total.

- [ ] **Step 4: Implement the thin H24 runner**

```python
H24 = 24
EFFECTIVE_POPULATION = 512
TOTAL_UPDATES = 128
CHECKPOINT_UPDATES = 16
TRANSITIONS_PER_UPDATE = EFFECTIVE_POPULATION * H24
TOTAL_STEPS = TOTAL_UPDATES * TRANSITIONS_PER_UPDATE
CHECKPOINT_INTERVAL = CHECKPOINT_UPDATES * TRANSITIONS_PER_UPDATE

def build_fresh_full_action_h24_kwargs(profile_name, reference_path, seed):
    kwargs = build_fresh_ppo_action_contract_kwargs(
        profile_name, reference_path, seed
    )
    kwargs.update(
        unroll_length=H24,
        total_steps=TOTAL_STEPS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )
    return kwargs

def expected_checkpoint_steps():
    return tuple(
        range(CHECKPOINT_INTERVAL, TOTAL_STEPS + 1, CHECKPOINT_INTERVAL)
    )
```

The CLI must reuse E021's code/reference preflight, add exact H24/cadence fields,
call `train` under the pinned solver context, and call the generalized validator
with the H24 kwargs, steps, total, and protocol.

- [ ] **Step 5: Verify GREEN and neighboring contracts**

Run:
`PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/pytest -q tests/test_g1_fresh_full_action_h24_walk.py tests/test_g1_fresh_ppo_action_contract_walk.py tests/test_g1_fresh_fixed_noise_training.py`

Expected: all tests pass.

- [ ] **Step 6: Run static checks and commit**

Run Ruff on the four changed files, `python -m py_compile` on both runners, and
`git diff --check`. Commit only the runner and test files.

### Task 2: Register, execute, and evaluate E022

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260814-022.yaml`
- Modify after evaluation: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generate after evaluation: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/Dashboard.md`

**Interfaces:**
- Consumes: the clean Task 1 code commit and E021's immutable baseline artifacts.
- Produces: one immutable E022 run, exact five-phase grids, clean/noisy videos, joint diagnostics, and a curated registry decision.

- [ ] **Step 1: Preregister E022**

Create one approved schema-v1 record with parent E021, the exact Task 1 commit,
clean patch hash, `conda:diffsim`, model/reference hashes, g1-4x5 solver, seed
zero, GPU count one, 90-minute budget, the design's six-outcome map, and these
required artifacts: preflight, training validation, eight checkpoint archives,
four five-phase grids, selected clean/noisy rollouts, contact sheet, learning
plot, PPO catch comparison, and selection JSON.

- [ ] **Step 2: Validate and dry-run**

Run:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260814-022 --dry-run
```

Expected: registry valid and the dry-run prints one seed-zero GPU command.

- [ ] **Step 3: Launch and monitor the guarded run**

Run `uv run python -m tools.runexp E-20260814-022`. At the first checkpoint,
require finite actor gradients/updates, all five occupied CAGrad bins, every
post-clip bin norm at most 1.0, action noise exactly 0.2, bootstrap exactly
zero, and a checkpoint step of 196,608. Stop only for an invalid-execution or
instability gate.

- [ ] **Step 4: Evaluate the registered checkpoints**

Evaluate updates 16/32/64/128 at phases 0/25/50/75/100 using the checkpoint
hparams action contract. For the componentwise-safe selected checkpoint,
produce clean and noisy phase-zero videos plus joint traces. Compare phases
40--61 with the fixed PPO control and E021 metrics.

- [ ] **Step 5: Curate and publish**

Apply exactly one preregistered outcome, update E022 and current state from
validated artifacts, copy MP4/PNG previews to the Obsidian attachment path,
run registry validation, rebuild the vault, check dashboard links, and commit
only owned canonical records.
