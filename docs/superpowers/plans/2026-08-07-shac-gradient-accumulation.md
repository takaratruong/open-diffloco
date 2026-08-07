# SHAC Gradient Accumulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add memory-bounded, optimizer-correct gradient accumulation to SHAC so 256-environment microbatches can test effective G1 batches of 512 and 1,024.

**Architecture:** Keep the complete effective environment population in `TrainState`, but reshape it into fixed physical microbatches and scan the existing per-environment differentiator sequentially. Apply finite filtering and per-environment clipping inside each shard, average shard gradients before one actor or critic optimizer update, then flatten trajectories and final states for the existing normalizer, metric, and checkpoint paths.

**Tech Stack:** Python 3.11, JAX/MJX, Flax, Optax, NumPy, `unittest`, existing Open-DiffLoco SHAC trainer and G1 RMR runner.

## Global Constraints

- `num_envs` remains the physical microbatch size; `gradient_accumulation_steps` is a positive integer defaulting to one.
- `effective_num_envs = num_envs * gradient_accumulation_steps` and one actor update advances `effective_num_envs * unroll_length` steps.
- Apply existing finite rejection and per-environment norm clipping before averaging shards; perform exactly one Adam actor update and one Adam critic update per critic iteration.
- Factor one preserves current behavior, compilation signature, step arithmetic, and checkpoint semantics.
- Generate ordered randomness over the full effective population before reshaping; never reuse shard randomness.
- Do not add host-side optimizer loops, direct larger `num_envs`, distributed execution, or generic orchestration.
- Preserve the validated fixed-scan MJX solver 4/5 and every E-20260805-098 task, model, reward, observation, initialization, and optimizer input.

---

### Task 1: Extend per-environment gradient summaries

**Files:**
- Modify: `src/algorithms/shac/gradients.py`
- Modify: `tests/test_shac_gradient_aggregation.py`

**Interfaces:**
- Consumes: a gradient pytree whose leaves share leading environment axis and `max_norm: float`.
- Produces: unchanged aggregated clipped mean plus stats containing scalar `finite_fraction`, `raw_norm_median`, `raw_norm_max` and vectors `finite_by_env`, `raw_norm_by_env`.

- [ ] **Step 1: Write failing full-population-stat tests**

Add tests that split the same six gradient rows into two three-row shards, concatenate `raw_norm_by_env` and `finite_by_env`, and assert their finite fraction, median, and maximum match an explicit six-row call. Also assert non-finite rows remain zero-weighted in the aggregated mean.

```python
def test_shard_statistics_reconstruct_full_population(self):
    gradients = {"w": jnp.array([[1., 0.], [2., 0.], [jnp.nan, 1.],
                                  [0., 4.], [0., 5.], [0., 6.]])}
    full, full_stats = aggregate_per_env_gradients(gradients, max_norm=2.0)
    shard_stats = [
        aggregate_per_env_gradients({"w": gradients["w"][start:start + 3]}, max_norm=2.0)[1]
        for start in (0, 3)
    ]
    norms = jnp.concatenate([stats["raw_norm_by_env"] for stats in shard_stats])
    finite = jnp.concatenate([stats["finite_by_env"] for stats in shard_stats])
    self.assertAlmostEqual(float(jnp.mean(finite)), float(full_stats["finite_fraction"]))
    self.assertAlmostEqual(float(jnp.median(norms)), float(full_stats["raw_norm_median"]))
    self.assertAlmostEqual(float(jnp.max(norms)), float(full_stats["raw_norm_max"]))
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_shac_gradient_aggregation -v
```

Expected: the new test fails with missing `raw_norm_by_env` or `finite_by_env`.

- [ ] **Step 3: Expose the two diagnostic vectors**

Add these keys without changing aggregation arithmetic:

```python
stats = {
    "finite_fraction": jp.mean(finite_by_env.astype(jp.float32)),
    "raw_norm_median": jp.median(raw_norm),
    "raw_norm_max": jp.max(raw_norm),
    "finite_by_env": finite_by_env,
    "raw_norm_by_env": raw_norm,
}
```

- [ ] **Step 4: Run tests and commit**

Run the Task 1 test command; expect all tests to pass. Commit:

```bash
git add src/algorithms/shac/gradients.py tests/test_shac_gradient_aggregation.py
git commit -m "feat: expose per-environment gradient diagnostics"
```

### Task 2: Add reusable microbatch tree operations

**Files:**
- Create: `src/algorithms/shac/microbatch.py`
- Create: `tests/test_shac_microbatch.py`

**Interfaces:**
- Produces `reshape_population(tree, accumulation_steps, microbatch_size)`, `flatten_population(tree)`, `mean_shard_trees(shard_trees)`, and `summarize_shard_stats(shard_stats)`.
- All helpers are pure JAX pytree transformations and introduce no optimizer or environment knowledge.

- [ ] **Step 1: Write failing helper tests**

Test round-trip order for nested pytrees shaped `(8, ...) -> (2, 4, ...) -> (8, ...)`; test the mean of three gradient trees; and test that shard statistics reconstruct explicit concatenated finite fraction, median, and maximum.

- [ ] **Step 2: Run the test and verify failure**

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_shac_microbatch -v
```

Expected: import failure because `src.algorithms.shac.microbatch` does not exist.

- [ ] **Step 3: Implement the pure helpers**

Use exact leading-axis transformations:

```python
def reshape_population(tree, accumulation_steps, microbatch_size):
    effective = accumulation_steps * microbatch_size
    leaves = jax.tree_util.tree_leaves(tree)
    if any(value.shape[0] != effective for value in leaves):
        raise ValueError("every population leaf must use the effective leading size")
    return jax.tree_util.tree_map(
        lambda value: value.reshape(
            (accumulation_steps, microbatch_size) + value.shape[1:]
        ),
        tree,
    )

def flatten_population(tree):
    return jax.tree_util.tree_map(
        lambda value: value.reshape((value.shape[0] * value.shape[1],) + value.shape[2:]),
        tree,
    )

def mean_shard_trees(shard_trees):
    return jax.tree_util.tree_map(lambda value: jp.mean(value, axis=0), shard_trees)
```

`summarize_shard_stats` flattens the two diagnostic axes, then computes one finite fraction, median, and maximum.

- [ ] **Step 4: Run both helper suites and commit**

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_shac_microbatch tests.test_shac_gradient_aggregation -v
git add src/algorithms/shac/microbatch.py tests/test_shac_microbatch.py
git commit -m "feat: add SHAC microbatch tree operations"
```

Expected: both suites pass.

### Task 3: Transport and validate accumulation configuration

**Files:**
- Modify: `tools/run_g1_tracking_rmr50_shac.py`
- Modify: `tests/test_g1_tracking_runner.py`
- Modify: `src/algorithms/shac/algorithm.py`

**Interfaces:**
- Runner accepts `--gradient-accumulation-steps` and `build_train_kwargs(..., gradient_accumulation_steps: int = 1)`.
- Trainer accepts `gradient_accumulation_steps: int = 1`, records it in hparams, and derives `effective_num_envs` and `steps_per_actor_update` once.

- [ ] **Step 1: Write failing validation and transport tests**

Assert factor four reaches trainer kwargs, default is one, and `0`, `-1`, `1.5`, and `True` raise `ValueError` mentioning `gradient_accumulation_steps`.

- [ ] **Step 2: Run the runner tests and verify failure**

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_runner -v
```

Expected: unexpected keyword or missing CLI transport failure.

- [ ] **Step 3: Add validation, CLI transport, and effective arithmetic**

Validate before `train()` creates its output directory:

```python
if (
    isinstance(gradient_accumulation_steps, bool)
    or not isinstance(gradient_accumulation_steps, int)
    or gradient_accumulation_steps < 1
):
    raise ValueError("gradient_accumulation_steps must be a positive integer")
effective_num_envs = num_envs * gradient_accumulation_steps
steps_per_actor_update = effective_num_envs * unroll_length
```

Use `steps_per_actor_update` for learning-rate iteration counts, state step,
training-loop iteration counts, and checkpoints. Reset `effective_num_envs`
independent environments. Record all three values in `hparams.json`.

- [ ] **Step 4: Run runner and checkpoint tests and commit**

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_tracking_runner tests.test_shac_checkpoint_archival -v
git add tools/run_g1_tracking_rmr50_shac.py tests/test_g1_tracking_runner.py \
  src/algorithms/shac/algorithm.py
git commit -m "feat: configure SHAC gradient accumulation"
```

Expected: all tests pass.

### Task 4: Accumulate actor and critic gradients before optimization

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Create: `tests/test_shac_accumulated_update.py`

**Interfaces:**
- Factor one uses the existing direct batch path.
- Factor greater than one scans equal-sized shards and returns flattened `trajs`, flattened `final_states`, full-population losses, one actor gradient tree, and full-population statistics.
- Each critic iteration similarly returns one accumulated gradient tree before one optimizer update.

- [ ] **Step 1: Add a failing optimizer-count and aggregation test**

Use a small pure quadratic loss with four examples split into two shards. Assert
the accumulated clipped gradient equals explicit full-batch aggregation and a
counting Optax transform observes one update, not two. Repeat for two simulated
critic iterations and expect exactly two updates.

- [ ] **Step 2: Run the test and verify failure**

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_shac_accumulated_update -v
```

Expected: failure because the trainer has no accumulated-update boundary.

- [ ] **Step 3: Implement the actor microbatch scan**

Generate randomness with leading `effective_num_envs`, reshape state and random
inputs with `reshape_population`, then scan a body that runs the current
per-environment `actor_grad_fn` and `aggregate_per_env_gradients`. Average the
returned shard gradient trees only after the scan. Flatten the returned
trajectories and final states in original population order before downstream
use. Keep the existing direct code in the static factor-one branch.

- [ ] **Step 4: Implement accumulated critic iterations**

Inside each critic iteration, reshape stored trajectory inputs into shards,
scan `jax.vmap(jax.value_and_grad(single_env_critic_loss))`, aggregate each
shard, average the shard means, and call `critic_opt.update` once. Preserve 16
outer critic iterations and one target update.

- [ ] **Step 5: Preserve combined normalizers and metrics**

Flatten all effective trajectories before the existing normalizer updates.
Compute metrics over every effective environment, preserve flattened final
states in `TrainState`, and advance by `steps_per_actor_update`.

- [ ] **Step 6: Run focused suites and commit**

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_shac_accumulated_update \
  tests.test_shac_microbatch \
  tests.test_shac_gradient_aggregation \
  tests.test_g1_tracking_runner \
  tests.test_shac_checkpoint_archival -v
git add src/algorithms/shac/algorithm.py tests/test_shac_accumulated_update.py
git commit -m "feat: accumulate SHAC gradients across microbatches"
```

Expected: all focused tests pass.

### Task 5: Build the real-G1 parity evaluator

**Files:**
- Create: `tools/validate_g1_shac_gradient_accumulation.py`
- Create: `tests/test_g1_shac_gradient_accumulation_validator.py`

**Interfaces:**
- CLI accepts `--output`, `--seed`, `--monolithic-envs`, `--microbatch-envs`, `--accumulation-steps`, and solver iterations.
- Runs one monolithic and one accumulated update under the fixed solver and writes strict finite `comparison.json` atomically.

- [ ] **Step 1: Write failing comparison-helper tests**

Test tree cosine, relative L2, exact step/shape checks, strict finite JSON, and
the registered thresholds: cosine at least `0.9999`, relative L2 at most
`1e-4`, normalizer relative error at most `1e-6`.

- [ ] **Step 2: Run the validator tests and verify failure**

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_shac_gradient_accumulation_validator -v
```

Expected: import failure because the validator does not exist.

- [ ] **Step 3: Implement the bounded validator**

Use the exact compact random-head validated-task kwargs. Run monolithic four
environments/factor one and microbatch two/factor two for one actor update each
from the same seed, compare returned state and optimizer pytrees, include run
directories and timings, reject non-finite values, and publish JSON with
`allow_nan=False` through an atomic rename.

- [ ] **Step 4: Run all focused tests and commit**

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_shac_gradient_accumulation_validator \
  tests.test_shac_accumulated_update \
  tests.test_shac_microbatch \
  tests.test_shac_gradient_aggregation \
  tests.test_g1_tracking_runner -v
git add tools/validate_g1_shac_gradient_accumulation.py \
  tests/test_g1_shac_gradient_accumulation_validator.py
git commit -m "feat: validate G1 SHAC accumulation parity"
git push takaratruong research/g1-rmr-50hz-20260805
```

Expected: all tests pass and the implementation branch is published.

### Task 6: Execute parity and production resource gates

**Files:**
- Create in research repository: `research/experiments/E-20260807-000.yaml`
- Create in research repository after parity passes: `research/experiments/E-20260807-001.yaml`

**Interfaces:**
- E000 validates monolithic-4 versus microbatch-2×2 actual MJX parity.
- E001 validates one full effective-512 update as microbatch-256×2 on one L40S.

- [ ] **Step 1: Preregister, validate, and dry-run E000**

Pin the clean implementation commit, model/reference hashes from E098, solver
4/5, seed zero, one GPU, 120-minute wall time, the exact validator command,
and parity/invalid outcome map. Run registry validation and guarded dry-run.

- [ ] **Step 2: Execute and curate E000**

Run the guarded experiment. Verify `comparison.json`, exact provenance,
finite leaves, cosine/error gates, and no abandoned process. If parity fails,
stop the branch and do not create E001.

- [ ] **Step 3: Preregister and execute E001 after parity passes**

Register one update with physical batch 256, factor two, horizon 12, total
steps 6,144, and every E098 setting. Require memory headroom, one actor plus
16 critic updates, finite diagnostics/checkpoints, exact hparams, and no second
real-step compile. Validate, dry-run, execute, and curate.

### Task 7: Run factor-2 and factor-4 learning arms

**Files:**
- Create in research repository: `research/experiments/E-20260807-002.yaml`
- Create in research repository: `research/experiments/E-20260807-003.yaml`

**Interfaces:**
- E002: physical 256, accumulation 2, effective 512, horizon 12, 128 actor updates, 786,432 transitions.
- E003: physical 256, accumulation 4, effective 1,024, horizon 12, 128 actor updates, 1,572,864 transitions.

- [ ] **Step 1: Preregister both arms**

Use E098 as immutable factor-one control. Pin identical seed, compact actor,
random head, no LayerNorm, actor LR `1e-3`, action noise `1.0`, bootstrap zero,
task, rewards, solver, fixed updates, and evaluation selection. The sole causal
change between arms is accumulation factor/effective batch.

- [ ] **Step 2: Validate, dry-run, and launch on separate GPUs**

After E001 passes, validate both registry records and dry-run exact commands.
Launch concurrently only because the user explicitly requested varied
accumulation amounts and the arms have isolated GPUs, caches, and run roots.

- [ ] **Step 3: Evaluate fixed checkpoints and apply the gate**

Evaluate updates 10/20/30/40/50/128 at phase zero, then selected phases
30/60/90 with strict summary, trajectory, MP4, and contact sheet. Retain the
smallest factor exceeding E098's 55/60 survival. Run factor eight only if
factor four exceeds 55; otherwise pivot away from batch variance.

- [ ] **Step 4: Curate and publish**

Update both experiment records, H-G1-028, current state, and generated Obsidian
pages from deterministic artifacts. Validate the registry, rebuild the vault,
commit only owned files, and push both research and implementation branches.
