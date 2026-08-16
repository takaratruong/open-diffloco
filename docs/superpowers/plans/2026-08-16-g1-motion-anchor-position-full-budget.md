# G1 Motion-Anchor Position Full-Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the reviewed E004 root-position runner with a fail-closed 128-update mode and execute one matched E005 comparison against E023.

**Architecture:** Keep one runner and one focused test module. Represent the two immutable execution budgets with a small frozen value object, thread the selected object through kwargs, preflight, artifact validation, and classification, and preserve the omitted/default invocation as the exact E004 early path. The full path changes no environment, trainer, objective, actor, evaluator, or simulator code.

**Tech Stack:** Python 3.11, pytest, Ruff, SHAC/JAX/MJX, `diffsim2real-lab` registry and `researchctl`.

## Global Constraints

- The only learning-semantic difference from E023 is `actor_observe_motion_anchor_position=True` with `expected_actor_obs_dim=3310`.
- Preserve ten-frame history, H24, 256 physical environments, two accumulation shards, effective population 512, five-bin CAGrad, per-environment clipping, zero terminal bootstrap, exact reference resets, nominal physics, no observation noise or pushes, seed zero, and the action-parity boundary.
- Preserve the RMR action-noise schedule endpoint at transition `1_572_864` in both budget modes.
- `early` remains the omitted/default mode with 32 updates, `393_216` transitions, and archives at `196_608/393_216`.
- `full` uses 128 updates, `1_572_864` transitions, and archives at `196_608/393_216/589_824/786_432/983_040/1_179_648/1_376_256/1_572_864`.
- Full execution is invalid unless update-16 and update-32 replay-free vectors exactly equal E004's `43/38/48/49/24` and `51/66/55/49/24`.
- Update 128 is the sole decisive and retention-eligible checkpoint; compare it to E023 update 128 `116/99/67/49/24`. Update 64 `71/60/51/49/24` is diagnostic.
- Keep E004 immutable and do not launch tuning, a resume, or another scientific treatment before E005 is evaluated and curated.

---

## File Map

- Modify `tools/run_g1_motion_anchor_position_h24_walk.py`: define and resolve the two immutable budget contracts, thread them through execution, and classify full-budget evidence.
- Modify `tests/test_g1_motion_anchor_position_h24_walk.py`: prove default compatibility, exact full deltas, JAX-free preflight, archive requirements, and all full outcomes.
- Create `research/experiments/E-20260816-005.yaml` in `/home/ubuntu/projects/diffsim2real-lab`: preregister the one full-budget run after the code commit is clean and reviewed.
- Generate only under `/home/ubuntu/projects/diffsim2real-lab/runs/E-20260816-005/`: canonical run, evaluation, media, and manifest-last evidence.
- Update registry-derived lab records and vault through `tools/researchctl.py`; never hand-edit generated dashboard pages.

### Task 1: Immutable early/full execution budget

**Files:**
- Modify: `tools/run_g1_motion_anchor_position_h24_walk.py`
- Modify: `tests/test_g1_motion_anchor_position_h24_walk.py`

**Interfaces:**
- Consumes: existing `build_rmr_noise_h24_kwargs`, `validate_e023_preflight`, and `validate_training_artifacts` seams.
- Produces: frozen `BudgetContract`, `EARLY_BUDGET`, `FULL_BUDGET`, `resolve_budget(name: str) -> BudgetContract`, and budget-aware `expected_checkpoint_steps`, `build_motion_anchor_position_kwargs`, and `validate_preflight` functions.

- [ ] **Step 1: Write failing budget tests**

Add tests that import `EARLY_BUDGET`, `FULL_BUDGET`, and `resolve_budget`; require exact update/transition/archive tuples; require `resolve_budget("early") is EARLY_BUDGET`, `resolve_budget("full") is FULL_BUDGET`, and rejection of unknown strings. Keep the existing builder test unchanged, then add a full-builder comparison:

```python
full = build_motion_anchor_position_kwargs("g1-4x5", reference, 0, budget=FULL_BUDGET)
changed = {
    key
    for key in set(parent) | set(full)
    if not np.array_equal(parent.get(key), full.get(key))
}
assert changed == {"actor_observe_motion_anchor_position", "expected_actor_obs_dim"}
assert full["total_steps"] == 1_572_864
assert expected_checkpoint_steps(FULL_BUDGET) == (
    196_608, 393_216, 589_824, 786_432,
    983_040, 1_179_648, 1_376_256, 1_572_864,
)
```

- [ ] **Step 2: Run the new budget slice and confirm RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_motion_anchor_position_h24_walk.py -k 'budget or full_builder'
```

Expected: collection or assertion failures because the budget APIs do not exist.

- [ ] **Step 3: Implement the minimal immutable contract**

Add a frozen dataclass with only the execution fields:

```python
@dataclass(frozen=True)
class BudgetContract:
    name: str
    total_updates: int
    checkpoint_updates: tuple[int, ...]

    @property
    def total_steps(self) -> int:
        return self.total_updates * TRANSITIONS_PER_UPDATE

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        return tuple(update * TRANSITIONS_PER_UPDATE for update in self.checkpoint_updates)
```

Instantiate `EARLY_BUDGET` for `(16, 32)` and `FULL_BUDGET` for `(16, 32, 48, 64, 80, 96, 112, 128)`. Retain the legacy module constants as aliases of `EARLY_BUDGET` so existing imports and E004 tests do not change. Add an explicit mapping-based resolver that raises `ValueError` on unknown input.

Make the three existing APIs accept `budget: BudgetContract = EARLY_BUDGET`; set only `total_steps=budget.total_steps`, and publish `budget.name`, `total_updates`, `checkpoint_updates`, and `checkpoint_steps` in preflight. Do not change imports that would initialize JAX.

- [ ] **Step 4: Add CLI and execution threading tests**

Require the parser default to be `early`, accept `--budget full`, and reject any other value. Monkeypatch the runner's trainer and validator seams or inspect the factored execution helper so the test proves full mode forwards all eight expected steps and `total_steps=1_572_864` to `validate_training_artifacts`.

- [ ] **Step 5: Thread the resolved budget through `main`**

Add:

```python
parser.add_argument("--budget", choices=("early", "full"), default="early")
```

Resolve it before preflight and pass the same object to `validate_preflight`, `build_motion_anchor_position_kwargs`, `expected_checkpoint_steps`, and the artifact validator. Use a budget-specific training protocol string so evidence from the early and full executions cannot be confused.

- [ ] **Step 6: Run focused GREEN and static checks**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_motion_anchor_position_h24_walk.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
  tools/run_g1_motion_anchor_position_h24_walk.py \
  tests/test_g1_motion_anchor_position_h24_walk.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
  tools/run_g1_motion_anchor_position_h24_walk.py
git diff --check
```

Expected: all focused tests pass and every static command exits zero.

- [ ] **Step 7: Commit Task 1**

```bash
git add tools/run_g1_motion_anchor_position_h24_walk.py \
  tests/test_g1_motion_anchor_position_h24_walk.py
git commit -m "feat: add full root-position training budget"
```

### Task 2: Full-budget scientific outcome contract

**Files:**
- Modify: `tools/run_g1_motion_anchor_position_h24_walk.py`
- Modify: `tests/test_g1_motion_anchor_position_h24_walk.py`

**Interfaces:**
- Consumes: Task 1 `FULL_BUDGET`, phase caps, E004 early vectors, and fixed E023 update-64/update-128 vectors.
- Produces: `classify_full_budget_root_position(treatment: Mapping[int, Sequence[int]]) -> str`, which is pure, deterministic, and fail-closed.

- [ ] **Step 1: Write failing classifier tests**

Require evidence keys exactly `{16, 32, 64, 128}`. Require update 16 and 32 to exactly match E004 before any later comparison. Add one explicit vector for each outcome:

```python
EARLY = {16: (43, 38, 48, 49, 24), 32: (51, 66, 55, 49, 24)}
assert classify_full_budget_root_position({
    **EARLY, 64: (71, 60, 51, 49, 24), 128: (117, 99, 67, 49, 24)
}) == "root-position-full-advances"
assert classify_full_budget_root_position({
    **EARLY, 64: (71, 60, 51, 49, 24), 128: (114, 99, 67, 49, 24)
}) == "root-position-full-parity"
assert classify_full_budget_root_position({
    **EARLY, 64: (71, 60, 51, 49, 24), 128: (120, 95, 67, 49, 24)
}) == "root-position-full-mixed"
assert classify_full_budget_root_position({
    **EARLY, 64: (71, 60, 51, 49, 24), 128: (110, 95, 64, 45, 24)
}) == "root-position-full-underperforms"
```

Also reject missing/extra updates, wrong early corroboration, non-integer/nonfinite values, wrong vector width, and phase-suffix overflow.

- [ ] **Step 2: Run the classifier tests and confirm RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_motion_anchor_position_h24_walk.py -k full_budget
```

Expected: failure because the full classifier does not exist.

- [ ] **Step 3: Implement the minimal pure classifier**

Validate all four rows against `PHASE_CAPS`; then compare only update 128 against `(116, 99, 67, 49, 24)`. Return advance when all five deltas are nonnegative and one of the first four is positive; otherwise parity when every absolute delta is at most two; otherwise mixed when some delta is greater than two and another is less than negative two; otherwise underperform. Raise `ValueError` rather than returning an outcome when early corroboration or evidence validation fails.

- [ ] **Step 4: Verify Task 2 and commit**

Run the full focused test module, Ruff, py_compile, and diff-check commands from Task 1. Then:

```bash
git add tools/run_g1_motion_anchor_position_h24_walk.py \
  tests/test_g1_motion_anchor_position_h24_walk.py
git commit -m "feat: classify full root-position evidence"
```

### Task 3: Review, verify, and preregister E005

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260816-005.yaml`
- Update: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`

**Interfaces:**
- Consumes: clean Task 1-2 code commit and the existing E004/E023 canonical evidence.
- Produces: one approved, immutable experiment card whose command includes `--budget full`.

- [ ] **Step 1: Request an independent code review**

Review the complete design-to-diff mapping, with particular attention to default E004 compatibility, exact E023 delta, JAX-free preflight, all-eight checkpoint validation, classifier thresholds, and CLI threading. Repair every Critical or Important issue with a new RED/GREEN test before proceeding.

- [ ] **Step 2: Run final code verification**

Run the focused module and directly neighboring runner/preflight/training-validator tests, Ruff on touched files, py_compile, and `git diff --check`. Confirm the code worktree is clean and push the reviewed branch. Record the exact commit SHA in E005.

- [ ] **Step 3: Create the E005 experiment card**

Copy the schema of E004, set parent to `E-20260816-004`, use the exact clean code commit, and register the single command:

```bash
python -m tools.run_g1_motion_anchor_position_h24_walk \
  --solver-profile g1-4x5 \
  --reference-path <PINNED_WALK_REFERENCE> \
  --seed 0 \
  --budget full \
  --output-root <RUN_DIRECTORY> \
  --code-commit <EXACT_CLEAN_COMMIT>
```

Record the eight archive steps, E004 early-correspondence vectors, E023 update-64/update-128 controls, outcome definitions, 128-update stop rule, required media, and manifest-last publication contract.

- [ ] **Step 4: Validate and dry-run the registry**

Run:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260816-005 --dry-run
```

Expected: registry validation succeeds, the dry run resolves one command with `--budget full`, and no GPU process starts.

- [ ] **Step 5: Commit the preregistration**

Commit only the E005 card and intentional state change with a message such as `research: preregister full root-position run`.

### Task 4: Execute, evaluate, and curate E005

**Files:**
- Generate: `/home/ubuntu/projects/diffsim2real-lab/runs/E-20260816-005/<RUN_ID>/...`
- Update through tooling: E005 result record, findings, evidence manifest, vault dashboard, and current state.

**Interfaces:**
- Consumes: approved E005 registry card and one available GPU.
- Produces: one complete, independently inspectable full-budget result and a dashboard-visible selected rollout.

- [ ] **Step 1: Launch only the registered experiment**

Run:

```bash
uv run python -m tools.runexp E-20260816-005
```

Monitor compile completion, update cadence, finite actor/critic gradients, five occupied CAGrad bins, exact action-noise schedule, and all eight checkpoint archives. Stop on any invalid provenance, nonfinite telemetry, wrong checkpoint step, or early-correspondence failure.

- [ ] **Step 2: Evaluate the registered checkpoints**

Evaluate updates 16 and 32 first and require exact E004 vectors. If they match, evaluate updates 64 and 128 at phases `0/25/50/75/100` under the replay-free nominal evaluator. Persist phase-grid JSON/NPZ artifacts and run the pure full classifier.

- [ ] **Step 3: Render and diagnose only update 128**

Render phase zero and publish the trajectory NPZ, summary JSON, MP4, contact sheet, and learning/action diagnostics. Include action mean, sampled action, effective action, clipping/saturation fractions, return/survival curves, and the exact selected checkpoint SHA. Do not use rendered media to alter the registered classifier.

- [ ] **Step 4: Publish manifest last and curate**

Hash every canonical artifact, write the evidence manifest atomically last, update E005's result/outcome, add or update the practical finding linking E004 and E023, rebuild the vault with `uv run python tools/researchctl.py build-vault`, and verify the newest dashboard entry is first with playable MP4 and valid links.

- [ ] **Step 5: Verify and commit the research result**

Run registry tests and validation, inspect the dashboard/media, confirm no unrelated dirty files are included, commit the E005 result, and push the lab repository. Stop after curation; do not launch a successor experiment in this plan.
