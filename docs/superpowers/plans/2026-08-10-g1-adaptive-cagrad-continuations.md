# G1 Adaptive-Sampling And CAGrad Continuations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two default-off SHAC continuation treatments—failure-adaptive reference-phase resets and five-bin CAGrad actor aggregation—then run both as matched exact continuations of E-20260810-006.

**Architecture:** Keep treatment mathematics in two focused pure modules. Adaptive state is a small failure-count EMA carried inside the batched G1 environment info so old `TrainState` checkpoints remain loadable; the environment consumes the completed prior update's distribution. CAGrad reduces per-environment gradients to five bin accumulators inside each physical shard, merges the two accumulators, and performs one fixed-size JAX simplex solve before the existing optimizer step.

**Tech Stack:** Python 3.11, JAX, Flax, Optax, MuJoCo MJX, NumPy, SciPy in tests only, unittest/pytest, Ruff, YAML research registry.

## Global Constraints

- Resume exact E006 checkpoint SHA-256 `1c1d657333e5ede6e402b494deb9d6e9d94771efa258497dc8d16a79b8905cef` at step `786432`.
- Preserve physical 256 environments, two sequential shards, effective batch 512, horizon 12, seed zero, solver `g1-4x5`, and every saved optimizer/model/environment field.
- Add exactly 64 actor updates and `393216` transitions, ending at step `1179648`.
- Keep each feature default-off and mutually exclusive with the other treatment and the older detached phase-weighting treatment.
- Adaptive sampling is a literal 50-percent uniform plus 50-percent failure-EMA mixture; it never consumes actual collapsed states.
- CAGrad uses five bins, `alpha=0.5`, 32 fixed Frank-Wolfe iterations, and actor gradients only.
- Preserve exact evaluation reset behavior and ordinary default SHAC behavior.
- Do not modify user-owned Obsidian settings or canvas files in the lab repository.

---

### Task 1: Literal-Mixture Adaptive Phase Sampler

**Files:**
- Modify: `src/envs/g1_tracking/training_distribution.py`
- Modify: `tests/test_g1_tracking_training_distribution.py`

**Interfaces:**
- Consumes: existing `PhaseSamplerState`, `init_phase_sampler`, `update_phase_sampler`, and `sample_training_phase`.
- Produces: `phase_sampling_probabilities(state: PhaseSamplerState, uniform_ratio: float = 0.5) -> jax.Array` with literal-mixture semantics.

- [ ] **Step 1: Write failing probability tests**

Add tests equivalent to:

```python
def test_failure_mixture_retains_literal_uniform_floor(self):
    state = PhaseSamplerState(failed_count=jnp.array([0.0, 4.0]))
    probabilities = np.asarray(
        phase_sampling_probabilities(state, uniform_ratio=0.5)
    )
    np.testing.assert_allclose(probabilities, np.array([0.25, 0.75]))

def test_zero_failure_mixture_is_exactly_uniform(self):
    state = PhaseSamplerState(failed_count=jnp.zeros(5))
    probabilities = np.asarray(
        phase_sampling_probabilities(state, uniform_ratio=0.5)
    )
    np.testing.assert_allclose(probabilities, np.full(5, 0.2))
```

- [ ] **Step 2: Run the focused tests and confirm the old pseudocount rule fails**

Run:

```bash
pytest -q tests/test_g1_tracking_training_distribution.py -k 'mixture or sampler'
```

Expected: the concentrated two-bin assertion fails because the current helper does not implement a literal 50/50 mixture.

- [ ] **Step 3: Implement the literal mixture**

Replace only the probability calculation with:

```python
def phase_sampling_probabilities(
    state: PhaseSamplerState, uniform_ratio: float = 0.5
) -> jax.Array:
    if not 0.0 <= uniform_ratio <= 1.0:
        raise ValueError("uniform_ratio must be in [0, 1]")
    failures = jp.maximum(jp.asarray(state.failed_count), 0.0)
    bin_count = failures.shape[0]
    uniform = jp.full_like(failures, 1.0 / float(bin_count))
    total = jp.sum(failures)
    failure_distribution = jp.where(total > 0.0, failures / total, uniform)
    return uniform_ratio * uniform + (1.0 - uniform_ratio) * failure_distribution
```

Keep sampling within the selected bin and clipping to `reference_length - 2` unchanged.

- [ ] **Step 4: Run the sampler suite**

Run:

```bash
pytest -q tests/test_g1_tracking_training_distribution.py
```

Expected: all tests pass; concentrated sampling remains biased while every bin retains probability at least `0.5 / bin_count` under the registered setting.

- [ ] **Step 5: Commit the pure adaptive sampler change**

```bash
git add src/envs/g1_tracking/training_distribution.py tests/test_g1_tracking_training_distribution.py
git commit -m "feat: make G1 failure sampling a literal mixture"
```

---

### Task 2: Pure Five-Bin CAGrad Reducer

**Files:**
- Create: `src/algorithms/shac/cagrad.py`
- Create: `tests/test_shac_cagrad.py`

**Interfaces:**
- Consumes: a PyTree of per-environment gradients and a vector of reference phases.
- Produces: `PhaseGradientAccumulator`, `accumulate_phase_gradients`, `merge_phase_gradient_accumulators`, `finalize_phase_gradients`, and `combine_cagrad`.

- [ ] **Step 1: Write failing accumulator and identical-gradient tests**

Use these public shapes:

```python
accumulator = accumulate_phase_gradients(
    {"w": per_env_gradients}, phases,
    phase_count=500, bin_count=5,
)
task_gradients, counts, valid = finalize_phase_gradients(accumulator)
result = combine_cagrad(task_gradients, alpha=0.5, iterations=32)
```

Tests must prove that two shard accumulators merge to the same sums, finite
counts, and task means as one concatenated population, and that five identical
task gradients yield a combined direction collinear with their uniform mean.

- [ ] **Step 2: Run the tests and confirm the module is absent**

```bash
pytest -q tests/test_shac_cagrad.py
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement bin accumulation**

Create a `NamedTuple` with PyTree fields:

```python
class PhaseGradientAccumulator(NamedTuple):
    sums: Any
    finite_counts: Any
    env_counts: jax.Array
```

For every gradient leaf, add finite values into a leading `bin_count` axis and
add one to the matching finite-count position. `env_counts` counts environments
per bin independently of parameter finiteness. Merging adds corresponding
trees and counts. Finalization divides each sum by its elementwise finite count,
emits zero only where that count is zero, and returns `valid = all(env_counts > 0)`.

- [ ] **Step 4: Write the failing SciPy-oracle test**

Generate seeded matrices `A`, form positive-semidefinite Gram matrices as
`A.T @ A`, and compare the JAX solver's objective with:

```python
scipy.optimize.minimize(
    objective,
    np.full(task_count, 1.0 / task_count),
    bounds=[(0.0, 1.0)] * task_count,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    method="SLSQP",
)
```

Require simplex error at most `1e-6`, finite output, and objective gap at most
`1e-4` for the frozen fixtures.

- [ ] **Step 5: Implement fixed Frank-Wolfe CAGrad**

Flatten task-gradient leaves only for Gram construction. Starting from uniform
weights, run 32 JAX `lax.fori_loop` iterations. At iteration `t`, compute the
objective gradient, choose the minimum-gradient simplex vertex, and update with
`gamma = 2 / (t + 2)`. Return a `CAGradResult` containing combined gradient,
weights, Gram matrix, cosine matrix, objective, dual gap, uniform/combined
cosine, and a finite/simplex validity flag. Use `eps=1e-8` and the exact formula
from the approved design.

- [ ] **Step 6: Run focused tests and static checks**

```bash
pytest -q tests/test_shac_cagrad.py
ruff check src/algorithms/shac/cagrad.py tests/test_shac_cagrad.py
python -m py_compile src/algorithms/shac/cagrad.py tests/test_shac_cagrad.py
```

Expected: all commands return zero.

- [ ] **Step 7: Commit the pure CAGrad component**

```bash
git add src/algorithms/shac/cagrad.py tests/test_shac_cagrad.py
git commit -m "feat: add fixed-size SHAC CAGrad reducer"
```

---

### Task 3: Adaptive Reset Integration And Checkpoint State

**Files:**
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tests/test_g1_tracking_environment.py`
- Create: `tests/test_shac_adaptive_phase_sampling.py`

**Interfaces:**
- Consumes: Task 1's sampler functions.
- Produces: trainer settings `adaptive_phase_sampling=False`, `adaptive_phase_uniform_ratio=0.5`, and `adaptive_phase_alpha=0.001`.

- [ ] **Step 1: Write failing environment reset tests**

Construct an adaptive G1 environment with a concentrated two-bin failure state.
Using fixed keys, assert that adaptive resets select the concentrated bin more
often than uniform resets while qpos/qvel perturbations, domain-randomization
fields, actor-history shape, and continuation RNG remain finite and valid.
Assert that `reset_at_phase` remains exact and independent of the sampler.

- [ ] **Step 2: Write failing trainer-state tests**

Test a pure helper with this contract:

```python
updated = update_adaptive_phase_state(
    failed_count=jnp.zeros(bin_count),
    transition_phases=jnp.array([[10, 11], [149, 150]]),
    terminals=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
    reference_length=500,
    alpha=0.001,
)
```

Require both terminal phases to contribute regardless of shard layout. Add a
checkpoint migration test showing an E006-shaped environment state without the
new info leaf gains a broadcast zero EMA without changing any pre-existing
leaf byte values.

- [ ] **Step 3: Add the optional environment reset seam**

When `adaptive_phase_sampling` is false, retain the existing integer-uniform
phase draw. When true, carry `phase_sampler_failed_count` in environment info,
derive probabilities with `uniform_ratio=0.5`, sample a categorical bin and a
uniform frame inside it, then run the unchanged noisy-reference and domain-
randomization reset path. `step` passes the current state's EMA into its reset;
`reset_at_phase` never consumes it.

- [ ] **Step 4: Update the EMA once per actor update**

Inside `actor_loss`, emit the pre-reset transition phase alongside `terminal`.
After both physical shards have been flattened, update one EMA from all
effective-512 transitions. Broadcast the completed EMA into every final
environment state before constructing the next `TrainState`. The next update,
not the current partially completed update, consumes it.

- [ ] **Step 5: Record and serialize adaptive telemetry**

Add hparams and diagnostic leaves for failure EMA, probabilities, terminal bin
counts, and minimum bin probability. Validate that all are finite, probabilities
sum to one, and minimum probability is at least `0.5 / bin_count - 1e-7`.

- [ ] **Step 6: Run integration tests**

```bash
pytest -q tests/test_g1_tracking_training_distribution.py tests/test_g1_tracking_environment.py tests/test_shac_adaptive_phase_sampling.py tests/test_shac_microbatch.py
ruff check src/envs/g1_tracking/training_distribution.py src/envs/g1_tracking/environment.py src/algorithms/shac/algorithm.py tests/test_shac_adaptive_phase_sampling.py
```

Expected: all commands return zero.

- [ ] **Step 7: Commit adaptive integration**

```bash
git add src/envs/g1_tracking/environment.py src/algorithms/shac/algorithm.py tests/test_g1_tracking_environment.py tests/test_shac_adaptive_phase_sampling.py
git commit -m "feat: integrate adaptive G1 reset sampling"
```

---

### Task 4: Effective-512 CAGrad Integration

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Create: `tests/test_shac_cagrad_integration.py`

**Interfaces:**
- Consumes: Task 2's accumulator and `combine_cagrad` APIs.
- Produces: trainer settings `actor_cagrad=False`, `actor_cagrad_alpha=0.5`, `actor_cagrad_iterations=32`, sharing existing `actor_phase_bin_count=5`.

- [ ] **Step 1: Write failing validation tests**

Inspect `train`'s signature and assert defaults are exactly false, 0.5, and 32.
Require errors when CAGrad is combined with adaptive sampling, detached
phase-robust weighting, a non-G1 environment, per-environment actor clipping,
or a bin count other than five in the immutable runner.

- [ ] **Step 2: Write the two-shard integration test**

Build two small synthetic per-environment gradient shards with all five bins.
Run the same reducer path used by `train_step` and compare its accumulator,
task gradients, weights, and combined gradient with one concatenated batch.
Require exact bin counts and numerical agreement within `1e-6`.

- [ ] **Step 3: Integrate CAGrad inside shard reduction**

Capture actor start phases before population reshaping. For each shard, reduce
`shard_per_env_grads` immediately into `PhaseGradientAccumulator`; do not stack
512 per-environment parameter-gradient trees. Sum the stacked accumulators over
the shard axis, finalize five task gradients, run `combine_cagrad`, and pass its
combined gradient to the unchanged Optax chain. Retain the existing critic
shard mean exactly.

- [ ] **Step 4: Add diagnostics and hparams**

Serialize bin counts, bin gradient norms, weights, Gram and cosine matrices,
objective, dual gap, uniform/CAGrad cosine, combined norm, and validity. Keep
all older diagnostic leaves unchanged when the option is false.

- [ ] **Step 5: Run CAGrad and neighboring SHAC tests**

```bash
pytest -q tests/test_shac_cagrad.py tests/test_shac_cagrad_integration.py tests/test_shac_phase_weighting.py tests/test_shac_microbatch.py tests/test_shac_gradient_aggregation.py
ruff check src/algorithms/shac/cagrad.py src/algorithms/shac/algorithm.py tests/test_shac_cagrad.py tests/test_shac_cagrad_integration.py
```

Expected: all commands return zero and default-off tests demonstrate the old
aggregation path remains selected.

- [ ] **Step 6: Commit CAGrad integration**

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_cagrad_integration.py
git commit -m "feat: integrate phase CAGrad into effective SHAC batches"
```

---

### Task 5: Immutable Continuation Runners

**Files:**
- Create: `tools/run_g1_adaptive_phase_continuation.py`
- Create: `tools/run_g1_cagrad_continuation.py`
- Create: `tests/test_g1_adaptive_phase_continuation_runner.py`
- Create: `tests/test_g1_cagrad_continuation_runner.py`

**Interfaces:**
- Consumes: `build_canonical_kwargs`, the exact E006 checkpoint path supplied by the registered command, and Tasks 3–4's trainer flags.
- Produces: two CLI runners accepting only solver profile, reference path, seed, output root, and required resume path.

- [ ] **Step 1: Write failing contract tests**

For each builder, compare against an effective-512 exact continuation built as:

```python
base = build_canonical_kwargs(
    "g1-4x5", reference, seed=0, resume_from=checkpoint
)
base.update(
    gradient_accumulation_steps=2,
    total_steps=1_179_648,
    checkpoint_interval=196_608,
)
```

The adaptive builder may differ only by its three adaptive settings. The
CAGrad builder may differ only by its three CAGrad settings and fixed five-bin
setting. Parser tests reject batch, horizon, update-count, learning-rate,
mixture, alpha, and iteration overrides.

- [ ] **Step 2: Implement the two runners**

Follow `tools/run_g1_effective512_shac.py` for JAX configuration, solver
context, output-root handling, and final path printing. Require `--resume-from`
and resolve it before calling `train`; expose no treatment hyperparameters.

- [ ] **Step 3: Run runner and resume tests**

```bash
pytest -q tests/test_g1_adaptive_phase_continuation_runner.py tests/test_g1_cagrad_continuation_runner.py tests/test_g1_effective512_shac_runner.py tests/test_canonical_g1_shac_runner.py
```

Expected: all tests pass and the parent effective-512 runner remains unchanged.

- [ ] **Step 4: Commit runners**

```bash
git add tools/run_g1_adaptive_phase_continuation.py tools/run_g1_cagrad_continuation.py tests/test_g1_adaptive_phase_continuation_runner.py tests/test_g1_cagrad_continuation_runner.py
git commit -m "feat: add paired E006 continuation runners"
```

---

### Task 6: Source Verification And Review

**Files:**
- Modify only files identified by failures causally related to Tasks 1–5.

- [ ] **Step 1: Run the focused treatment suite in parallel processes**

```bash
pytest -q tests/test_g1_tracking_training_distribution.py tests/test_shac_adaptive_phase_sampling.py tests/test_g1_adaptive_phase_continuation_runner.py &
pytest -q tests/test_shac_cagrad.py tests/test_shac_cagrad_integration.py tests/test_g1_cagrad_continuation_runner.py &
wait
```

Expected: both processes return zero.

- [ ] **Step 2: Run neighboring regression tests and static checks**

```bash
pytest -q tests/test_g1_tracking_environment.py tests/test_shac_microbatch.py tests/test_shac_phase_weighting.py tests/test_shac_gradient_aggregation.py tests/test_canonical_g1_shac_runner.py tests/test_g1_effective512_shac_runner.py
ruff check src/algorithms/shac src/envs/g1_tracking tools/run_g1_adaptive_phase_continuation.py tools/run_g1_cagrad_continuation.py tests/test_shac_cagrad.py tests/test_shac_cagrad_integration.py tests/test_shac_adaptive_phase_sampling.py
python -m compileall -q src/algorithms/shac src/envs/g1_tracking tools
git diff --check
```

Expected: every command returns zero.

- [ ] **Step 3: Run one-update guarded smokes separately**

Use temporary output roots and the exact E006 checkpoint. Override the final
step only through a test-only Python call to each builder, setting it to one
additional effective update (`792576`). Require finite actor and critic
gradients, exact restore step, complete treatment telemetry, and a saved
checkpoint. Do not interpret behavior from these smokes.

- [ ] **Step 4: Request independent code review**

Review default-off behavior, E006 resume identity, reset distribution math,
terminal-phase timing, cross-shard accumulator correctness, CAGrad formula,
nonfinite handling, telemetry, and runner immutability. Repair Critical or
Important findings through focused failing tests before proceeding.

- [ ] **Step 5: Commit any review repairs**

```bash
git add src/algorithms/shac/algorithm.py src/algorithms/shac/cagrad.py src/envs/g1_tracking/environment.py src/envs/g1_tracking/training_distribution.py tests/test_shac_cagrad.py tests/test_shac_cagrad_integration.py tests/test_shac_adaptive_phase_sampling.py tests/test_g1_tracking_environment.py
git commit -m "fix: harden adaptive and CAGrad continuations"
```

Skip this commit when review finds no changes.

---

### Task 7: Preregister And Launch Paired Experiments

**Files in `/home/ubuntu/projects/diffsim2real-lab`:**
- Create: `research/experiments/E-20260810-007.yaml`
- Create: `research/experiments/E-20260810-008.yaml`
- Modify after validated results only: `research/state/current.yaml`
- Modify after validated results only: `research/hypotheses/H-SCI-001.yaml`

**Interfaces:**
- Consumes: one clean source commit containing both reviewed treatments and exact checkpoint SHA/path from E006.
- Produces: two separately approved, guarded, independently evaluable experiment records.

- [ ] **Step 1: Write the adaptive experiment record**

Register E007 as an exact E006 continuation changing only adaptive phase
sampling. Require finite final state, exact 64-update arithmetic, complete
sampler telemetry, fixed checkpoints at continuation updates 32/64, and the
shared phase-grid/nonregression gate from the design.

- [ ] **Step 2: Write the CAGrad experiment record**

Register E008 as an exact E006 continuation changing only actor CAGrad.
Require five occupied bins at every update, finite simplex/Gram telemetry,
fixed checkpoints at updates 32/64, and the identical behavioral gate.

- [ ] **Step 3: Validate and dry-run both contracts**

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260810-007 --dry-run
uv run python -m tools.runexp E-20260810-008 --dry-run
```

Expected: registry validation and both dry runs return zero, pin the same source
commit/checkpoint/model/reference/solver/environment, use distinct free GPUs,
and write to separate run directories.

- [ ] **Step 4: Commit registration without user Obsidian files**

```bash
git add research/experiments/E-20260810-007.yaml research/experiments/E-20260810-008.yaml
git commit -m "research: preregister adaptive and CAGrad continuations"
```

- [ ] **Step 5: Launch both guarded runs concurrently**

```bash
uv run python -m tools.runexp E-20260810-007
uv run python -m tools.runexp E-20260810-008
```

Launch through separate managed exec sessions so both run concurrently. Never
start an unregistered retry or silently change GPU, checkpoint, budget, or
settings.

---

### Task 8: Deterministic Evaluation And Curation

**Files in `/home/ubuntu/projects/diffsim2real-lab`:**
- Modify: `research/experiments/E-20260810-007.yaml`
- Modify: `research/experiments/E-20260810-008.yaml`
- Modify: `research/state/current.yaml`
- Modify only if evidence changes it: `research/hypotheses/H-SCI-001.yaml`
- Create only if promotion criteria are met: `docs/vault/findings/F-20260810-002 Adaptive Sampling Or CAGrad Raises The G1 Robust Floor.md`

- [ ] **Step 1: Verify immutable execution evidence**

Check return code, timeout, exact source/checkpoint/reference/model/solver hashes,
64-update and transition arithmetic, complete final TrainState finiteness,
required treatment telemetry, and both fixed checkpoints before behavioral
interpretation.

- [ ] **Step 2: Evaluate the fixed checkpoint grid**

For each arm, evaluate only continuation updates 32 and 64 at exact phases
`0/100/200/300/400`. Select lexicographically by minimum, median, mean, then
earliest checkpoint. Require trajectory arrays, finite summaries, videos, and
frame montages for any locomotion claim.

- [ ] **Step 3: Apply the preregistered decision map**

Advance an arm only if selected minimum is at least 19, median at least 48, and
every phase is at least E006 minus five frames. If one advances, retain only it.
If both advance, register a separate combined successor. If neither advances,
stop both and return to representation or longer credit assignment.

- [ ] **Step 4: Curate registry and Obsidian projection**

```bash
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py build-vault
pytest -q tests/test_research_registry.py tests/test_vault_projection.py
git diff --check
```

Expected: all commands return zero. Stage only experiment/hypothesis/current
state and generated projection files owned by this result; preserve the user's
existing `.obsidian` and canvas modifications.

- [ ] **Step 5: Commit and push the curated result**

```bash
git add research/experiments/E-20260810-007.yaml research/experiments/E-20260810-008.yaml research/state/current.yaml research/hypotheses/H-SCI-001.yaml docs/vault/experiments docs/vault/hypotheses docs/vault/Current\ State.md
git commit -m "research: curate adaptive and CAGrad continuations"
git push origin main
```

Remove from the `git add` command any unchanged or nonexistent path rather than
adding unrelated files.
