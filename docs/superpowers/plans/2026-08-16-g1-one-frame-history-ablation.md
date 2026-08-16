# G1 One-Frame History Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a bounded, provenance-safe E023 history ablation that changes the fresh H24 actor from ten 328-value frames to one and reports matched update-16/32 replay-free behavior.

**Architecture:** Generalize the existing Flax phase-grid evaluator so it records and uses the checkpoint's actual history length. Add one experiment-specific runner that derives E023 kwargs, changes only `actor_history_len`, shortens the execution budget to two existing checkpoint cadences without changing noise annealing, and provides a pure matched-control classifier. Use the existing trainer and renderers; do not change environment observations or generic SHAC learning.

**Tech Stack:** Python 3.11, JAX/MJX, Flax, NumPy, pytest, Ruff, existing DiffSim2Real registry/runexp tooling.

## Global Constraints

- Root-position features and every other observation-field change are excluded.
- The treatment actor input is exactly one 328-value frame.
- Preserve H24, effective population 512, five-bin CAGrad, actor LR `1e-3`, zero bootstrap, exact resets, nominal physics, and the pinned RMR action-noise schedule ending at transition `1,572,864`.
- Stop training after transition `393,216` (update 32), with archives at `196,608` and `393,216`.
- GPU evidence is interpretable only after clean-source preflight, finite training validation, matched compiled phase grids, video/contact sheet, and manifest-last artifact validation.
- Do not alter E023 or unrelated dirty files.

---

### Task 1: Make Flax phase-grid evidence history-faithful

**Files:**
- Modify: `tools/evaluate_g1_flax_phase_grid.py`
- Modify: `tests/test_evaluate_g1_flax_phase_grid.py`

**Interfaces:**
- Consumes: checkpoint sibling `hparams.json` with `actor_history_len`.
- Produces: `build_payload(..., actor_history_len: int, ...)` and evaluation calls that use the loaded contract history length.

- [ ] **Step 1: Write failing payload and action-path tests**

Add tests that call `build_payload` with `actor_history_len=1` and assert the JSON reports one, and inspect/call the evaluator seam so `evaluate_actor_action(..., history_len=1)` is used rather than the module constant. Also retain a ten-frame case.

```python
payload = build_payload(
    results=_five_results(),
    phases=(0, 25, 50, 75, 100),
    reference_transitions=124,
    checkpoint_path="checkpoint.pkl",
    checkpoint_sha256="a" * 64,
    reference_path="walk.npz",
    reference_sha256="b" * 64,
    solver_profile="g1-4x5",
    actor_reference_preview_mode="delta",
    actor_history_len=1,
)
assert payload["actor_history_len"] == 1
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_evaluate_g1_flax_phase_grid.py
```

Expected: FAIL because `build_payload` does not accept `actor_history_len` and currently publishes the constant ten.

- [ ] **Step 3: Implement the minimal dynamic-history wiring**

Change the payload signature and both residual/plain action calls:

```python
def build_payload(..., actor_history_len: int, ...) -> dict[str, object]:
    if isinstance(actor_history_len, bool) or actor_history_len < 1:
        raise ValueError("actor history length must be positive")
    return {
        ...,
        "actor_history_len": actor_history_len,
    }
```

In `main`, pass `contract["actor_history_len"]` to `build_payload`,
`evaluate_actor_action`, and `evaluate_gated_actor_action`.

- [ ] **Step 4: Run focused and neighboring tests**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_evaluate_g1_flax_phase_grid.py \
  tests/test_g1_tracking_evaluator.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add tools/evaluate_g1_flax_phase_grid.py \
  tests/test_evaluate_g1_flax_phase_grid.py
git commit -m "fix: bind Flax phase grids to actor history"
```

---

### Task 2: Add the bounded one-frame training contract

**Files:**
- Create: `tools/run_g1_one_frame_rmr_noise_h24_walk.py`
- Create: `tests/test_g1_one_frame_rmr_noise_h24_walk.py`

**Interfaces:**
- Consumes: `build_rmr_noise_h24_kwargs`, E023 constants, existing trainer/preflight/artifact validator.
- Produces: `build_one_frame_kwargs`, `expected_checkpoint_steps`, `classify_history_ablation`, parser, preflight, and executable `main`.

- [ ] **Step 1: Write failing runner-contract tests**

Test the wished-for API:

```python
def test_builder_changes_only_history_and_execution_budget():
    parent = build_rmr_noise_h24_kwargs("g1-4x5", Path("/tmp/walk.npz"), 0)
    treatment = build_one_frame_kwargs("g1-4x5", Path("/tmp/walk.npz"), 0)
    changed = {
        key for key in parent
        if not np.array_equal(parent[key], treatment[key])
    }
    assert changed == {"actor_history_len", "total_steps"}
    assert treatment["actor_history_len"] == 1
    assert treatment["total_steps"] == 393_216
    assert treatment["checkpoint_interval"] == 196_608
    assert treatment["action_noise_schedule_steps"] == 1_572_864
    assert expected_checkpoint_steps() == (196_608, 393_216)
```

Add exhaustive classifier tests for advances, parity, mixed, underperforms,
invalid shapes/nonfinite values, and phase-100 preservation.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_one_frame_rmr_noise_h24_walk.py
```

Expected: collection FAIL because the runner module is absent.

- [ ] **Step 3: Implement the minimal runner and classifier**

Use these immutable constants and builder:

```python
TOTAL_UPDATES = 32
TOTAL_STEPS = 393_216
CHECKPOINT_INTERVAL = 196_608
CONTROL = {
    16: (42, 36, 48, 49, 24),
    32: (45, 50, 53, 49, 24),
}

def expected_checkpoint_steps() -> tuple[int, ...]:
    return (CHECKPOINT_INTERVAL, TOTAL_STEPS)

def build_one_frame_kwargs(profile_name, reference_path, seed):
    kwargs = build_rmr_noise_h24_kwargs(profile_name, reference_path, seed)
    kwargs.update(actor_history_len=1, total_steps=TOTAL_STEPS)
    return kwargs
```

`classify_history_ablation` must validate exact update keys `(16, 32)`, five
finite integer survival values per row, and return outcomes in the exact
precedence defined by the design. `main` must write preflight before importing
JAX execution, train under the pinned solver context, validate exactly two
archives, and atomically write `training_validation.json`.

- [ ] **Step 4: Run runner, parent, and validator tests**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_one_frame_rmr_noise_h24_walk.py \
  tests/test_g1_rmr_noise_h24_walk.py \
  tests/test_g1_fresh_ppo_action_contract_walk.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add tools/run_g1_one_frame_rmr_noise_h24_walk.py \
  tests/test_g1_one_frame_rmr_noise_h24_walk.py
git commit -m "feat: add one-frame G1 history ablation"
```

---

### Task 3: Verify, register, and execute the bounded experiment

**Files:**
- Create in lab repo: `research/experiments/E-20260816-003.yaml`
- Generated after curation: `docs/vault/experiments/E-20260816-003.md`

**Interfaces:**
- Consumes: clean code commit from Tasks 1-2, E023 reference/model/controller hashes, E023 update-16/32 controls.
- Produces: approved registry record, immutable run artifacts, deterministic outcome, selected video/plot, and curated state.

- [ ] **Step 1: Run complete code verification**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_one_frame_rmr_noise_h24_walk.py \
  tests/test_g1_rmr_noise_h24_walk.py \
  tests/test_evaluate_g1_flax_phase_grid.py \
  tests/test_g1_tracking_evaluator.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
  tools/run_g1_one_frame_rmr_noise_h24_walk.py \
  tools/evaluate_g1_flax_phase_grid.py \
  tests/test_g1_one_frame_rmr_noise_h24_walk.py \
  tests/test_evaluate_g1_flax_phase_grid.py
python -m py_compile tools/run_g1_one_frame_rmr_noise_h24_walk.py \
  tools/evaluate_g1_flax_phase_grid.py
git diff --check
```

Expected: all tests and static checks pass and the code worktree is clean after commits.

- [ ] **Step 2: Register the exact one-frame experiment**

Write schema-v1 `E-20260816-003.yaml` with parent `E-20260814-023`, thread
`g1-tracking-shac`, tags `short-horizon-credit`, `closed-loop-transfer`, and
methods `mjx`, `shac`, `cagrad`, exact clean code commit, dirty SHA of the empty
patch, seed zero, one GPU, 60-minute budget, the five registered outcomes, and
required training/phase-grid/media/manifest artifacts.

- [ ] **Step 3: Validate and dry-run**

Run in the lab repo:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260816-003 --dry-run
```

Expected: registry valid and the dry run resolves the clean code commit,
reference, environment, command, output root, seed, and GPU without launching.

- [ ] **Step 4: Launch exactly one guarded experiment**

```bash
uv run python -m tools.runexp E-20260816-003
```

Expected: exactly two finite checkpoints and `training_validation.json`; no
other experiment starts.

- [ ] **Step 5: Evaluate and render both checkpoints**

Run the history-faithful Flax phase-grid evaluator for phases
`0 25 50 75 100` on both archives, classify the two vectors, select by the
registered first-four-phase key, and render the selected phase-zero checkpoint
with `tools/evaluate_g1_tracking.py` using its sibling hparams contract. Create
the learning diagnostic from the two checkpoint telemetry rows and publish the
manifest only after independently reopening every JSON/NPZ/MP4/PNG artifact.

- [ ] **Step 6: Curate and verify the result**

Update E003 and `research/state/current.yaml`, rebuild the vault, publish its
PNG/MP4 preview, and run:

```bash
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py build-vault
uv run pytest -q tests/test_registry.py
git diff --check
```

Expected: registry valid, dashboard rebuilt with E003 first, registry tests
pass, and the outcome authorizes either one full-budget continuation or the
root-position experiment—never both without a new record.
