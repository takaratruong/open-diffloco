# Fresh Noisy-RSI Walk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and launch one fresh H24 SHAC experiment that reproduces E023 while changing only reference-reset noise from scale 0 to the standard RMR scale 1 envelope.

**Architecture:** Add one experiment-specific runner that derives all training kwargs from `build_rmr_noise_h24_kwargs`, overwrites only `reference_reset_noise_scale`, and reuses the existing preflight and training-artifact validators. Register the clean runner commit in the plain-file research registry and execute it through `tools.runexp`; no algorithm or environment implementation changes are needed.

**Tech Stack:** Python 3.11, JAX/MJX, SHAC, pytest, Ruff, YAML experiment registry, `uv`.

## Global Constraints

- Start from a fresh zero-output actor, critic, optimizer, and normalizer; `resume_from` must remain absent.
- Use 128 H24 updates: `256` physical environments, gradient accumulation `2`, and `1_572_864` total transitions.
- Change only `reference_reset_noise_scale` from `0.0` to `1.0` relative to E023.
- Preserve seed 0, five-bin CAGrad, zero terminal bootstrap, the RMR action-noise schedule, the `g1-4x5` solver, plant, rewards, observations, and termination.
- Keep observation noise, domain randomization, pushes, assistance, adaptive phase sampling, and carried-state resets disabled.
- Archive and evaluate updates 16, 32, 64, and 128 at phases 0, 25, 50, 75, and 100.
- Do not launch until E024 is curated and the new registry record passes validation and dry run.

---

### Task 1: Fresh noisy-RSI runner

**Files:**
- Create: `tools/run_g1_fresh_noisy_rsi_h24_walk.py`
- Create: `tests/test_g1_fresh_noisy_rsi_h24_walk.py`

**Interfaces:**
- Consumes: `tools.run_g1_rmr_noise_h24_walk.build_rmr_noise_h24_kwargs`, `validate_preflight`, `TOTAL_STEPS`, `expected_checkpoint_steps`, and the existing SHAC artifact validator.
- Produces: `build_fresh_noisy_rsi_h24_kwargs(profile_name: str, reference_path: str | Path, seed: int) -> dict[str, Any]`, `validate_preflight(...) -> dict[str, Any]`, and a CLI compatible with the registry runner.

- [ ] **Step 1: Write the failing single-delta and preflight tests**

```python
def test_builder_changes_only_reference_reset_noise() -> None:
    baseline = build_rmr_noise_h24_kwargs("g1-4x5", Path("/tmp/walk.npz"), 0)
    treatment = build_fresh_noisy_rsi_h24_kwargs(
        "g1-4x5", Path("/tmp/walk.npz"), 0
    )
    changed = {
        key
        for key in set(baseline) | set(treatment)
        if not np.array_equal(baseline.get(key), treatment.get(key))
    }
    assert changed == {"reference_reset_noise_scale"}
    assert treatment["reference_reset_noise_scale"] == 1.0
    assert treatment.get("resume_from") is None
    assert treatment["domain_randomization"] is False
    assert treatment["actor_observation_noise"] is False
    assert treatment["push_velocity_range"] == (0.0, 0.0)


def test_preflight_records_fresh_noisy_rsi_only(monkeypatch) -> None:
    monkeypatch.setattr(runner, "validate_e023_preflight", lambda **_: {"valid": True})
    report = runner.validate_preflight(
        repository=Path("/repo"),
        reference_path=Path("/tmp/walk.npz"),
        code_commit="abc",
    )
    assert report["scientific_delta"] == ["reference_reset_noise_scale"]
    assert report["reference_reset_noise_scale"] == 1.0
    assert report["fresh_initialization"] is True
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest -q tests/test_g1_fresh_noisy_rsi_h24_walk.py
```

Expected: collection fails with `ModuleNotFoundError: tools.run_g1_fresh_noisy_rsi_h24_walk`.

- [ ] **Step 3: Implement the minimal derived runner**

```python
def build_fresh_noisy_rsi_h24_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    kwargs = build_rmr_noise_h24_kwargs(profile_name, reference_path, seed)
    kwargs["reference_reset_noise_scale"] = 1.0
    return kwargs


def validate_preflight(
    *, repository: Path, reference_path: Path, code_commit: str
) -> dict[str, Any]:
    base = validate_e023_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    return {
        **base,
        "protocol": "g1-fresh-noisy-rsi-h24-walk-preflight-v1",
        "scientific_delta": ["reference_reset_noise_scale"],
        "reference_reset_noise_scale": 1.0,
        "fresh_initialization": True,
    }
```

The CLI must write `preflight.json`, run `train(**kwargs)` inside the pinned solver context, validate the eight E023 checkpoint steps with `validate_training_artifacts`, and atomically write `training_validation.json`.

- [ ] **Step 4: Run focused and neighbor tests**

Run:

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest -q \
  tests/test_g1_fresh_noisy_rsi_h24_walk.py \
  tests/test_g1_rmr_noise_h24_walk.py \
  tests/test_g1_tracking_environment.py
/home/ubuntu/miniconda3/envs/rl/bin/python -m ruff check \
  tools/run_g1_fresh_noisy_rsi_h24_walk.py \
  tests/test_g1_fresh_noisy_rsi_h24_walk.py
python -m py_compile tools/run_g1_fresh_noisy_rsi_h24_walk.py
git diff --check
```

Expected: all tests and static checks pass; the environment tests prove the scale-one perturbation bounds and exact scale-zero path.

- [ ] **Step 5: Commit the runner**

```bash
git add tools/run_g1_fresh_noisy_rsi_h24_walk.py tests/test_g1_fresh_noisy_rsi_h24_walk.py
git commit -m "feat: add fresh noisy-RSI H24 walk"
```

### Task 2: Curate E024 and register the fresh treatment

**Files:**
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260814-023.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260814-024.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260814-025.yaml`

**Interfaces:**
- Consumes: E023 phase grid `116/99/67/49/24`; E024 continuation phase grids `69/99/67/49/24`, `105/99/74/49/24`, `74/99/74/49/24`, and `102/65/74/49/24`; the clean Task 1 commit.
- Produces: one evaluated E024 outcome, one approved E025 registry contract, and an updated current decision boundary.

- [ ] **Step 1: Curate E023 and E024 from deterministic evidence**

Set E023 to evaluated/pass with `rmr-noise-advances`. Set E024 to evaluated/fail with `continued-training-plateaus`, recording all four phase vectors and retaining E023 because no continuation checkpoint clears `116/99/67/49/24` componentwise. Update `current.yaml` to name noisy RSI as the next single causal treatment.

- [ ] **Step 2: Register E025**

Create `E-20260814-025.yaml` with parent E023, the clean Task 1 commit, empty dirty-patch SHA-256, pinned model/reference/solver/environment, seed 0, one GPU, 90-minute budget, and this command:

```bash
CODE_COMMIT=$(git -C /home/ubuntu/worktrees/open-diffloco/g1-rmr-action-parity-explicit-20260812 rev-parse HEAD)
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  tools/run_g1_fresh_noisy_rsi_h24_walk.py \
  --solver-profile g1-4x5 \
  --reference-path /home/ubuntu/projects/diffsim2real/outputs/rmr_motion_walk_win137_212_named.npz \
  --seed "$DIFFSIM_SEED" \
  --code-commit "$CODE_COMMIT" \
  --output-root "$DIFFSIM_SEED_DIR/training_runs"
```

The outcome map must distinguish `noisy-rsi-solves-walk`, `noisy-rsi-advances`, `noisy-rsi-insufficient`, `noisy-rsi-unstable`, and `invalid-execution`. Eligibility is componentwise `116/99/67/49/24`; completion is `124/99/74/49/24`.

- [ ] **Step 3: Validate and dry-run**

Run:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260814-025 --dry-run
```

Expected: registry validation passes and dry-run prints exactly one seed-zero GPU command with the pinned clean commit.

- [ ] **Step 4: Commit only owned registry files**

```bash
git add \
  research/experiments/E-20260814-023.yaml \
  research/experiments/E-20260814-024.yaml \
  research/experiments/E-20260814-025.yaml \
  research/state/current.yaml
git commit -m "research: register fresh noisy-RSI walk"
```

### Task 3: Guarded launch and evaluation checkpoint

**Files:**
- Create through the runner: the timestamped execution tree under `/home/ubuntu/projects/diffsim2real-lab/runs/E-20260814-025/`

**Interfaces:**
- Consumes: approved E025 and the pinned clean Task 1 code commit.
- Produces: immutable training checkpoints, validation telemetry, phase grids, learning curves, and selected rollout evidence.

- [ ] **Step 1: Launch only E025**

Run:

```bash
uv run python -m tools.runexp E-20260814-025
```

Expected: preflight records fresh initialization and exactly one scientific delta, then one GPU begins the 128-update training run.

- [ ] **Step 2: Monitor without changing the contract**

Require finite CAGrad, five occupied bins, nonzero finite actor gradients and updates, per-environment post-clip norm at most 1.0, exact scheduled action noise, zero bootstrap, and checkpoint steps 196608, 393216, 589824, 786432, 983040, 1179648, 1376256, and 1572864.

- [ ] **Step 3: Evaluate and curate before another experiment**

Evaluate updates 16, 32, 64, and 128 at the registered five phases. Generate learning/action diagnostics plus clean and noisy training-distribution videos only for the selected eligible checkpoint. Update the experiment result, current state, finding if promotion criteria are met, rebuild the vault, and validate before choosing carried-state RSI.
