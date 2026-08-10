# G1 Phase-Robust Actor Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, detached phase-bin weighting to the canonical G1 SHAC actor-gradient aggregation and run one bounded experiment against E-20260810-004.

**Architecture:** Keep uniform reference-state resets and all E004 scientific settings unchanged. A focused pure JAX module computes five-bin detached actor-loss weights and a finite-aware weighted gradient mean; `train` uses it only behind a disabled-by-default flag, and an immutable runner enables exactly that flag for the successor experiment.

**Tech Stack:** Python 3.11, JAX 0.4.38, MuJoCo/MJX 3.9.0, Flax 0.10.2, Optax 0.2.5, pytest, Ruff, YAML experiment registry, `tools.runexp`.

## Global Constraints

- Parent experiment: `E-20260810-004`; fresh seed 0; exactly 393,216 transitions.
- Preserve 256 environments, horizon 24, fixed-scan `g1-4x5`, actor and critic architecture, optimizer, rewards, bootstrap, action semantics, noise, domain randomization, curriculum, and checkpoint interval.
- Keep reset phases uniformly sampled. Do not invoke adaptive phase sampling or resume a checkpoint.
- Change only relative per-environment actor-gradient contributions; critic training remains uniform.
- Use five bins over 499 valid start phases and robust fraction `0.5`.
- Raw occupied-bin weights must be in `(0.5, 3.0]`; environment weights must have mean one and be detached.
- Nonfinite gradient elements contribute to neither weighted numerator nor denominator; an empty denominator produces zero.
- The established unweighted gradient path remains the default and retains its existing implementation.
- Stop unless execution is valid, minimum five-phase survival is at least 19, and median survival is at least 48.
- Never bind any service to `0.0.0.0`.

---

### Task 1: Pure phase-weighting and finite-aware aggregation

**Files:**
- Create: `src/algorithms/shac/phase_weighting.py`
- Create: `tests/test_shac_phase_weighting.py`

**Interfaces:**
- Produces: `phase_bin_indices(phases, *, phase_count, bin_count) -> jax.Array`.
- Produces: `phase_robust_weights(losses, phases, *, phase_count, bin_count, robust_fraction) -> PhaseWeighting`.
- Produces: `aggregate_phase_weighted_gradients(per_env_grads, env_weights) -> PyTree`.
- `PhaseWeighting` contains `bin_counts`, `bin_losses`, `bin_weights`, `env_weights`, `unweighted_loss`, `weighted_loss`, and `valid`.

- [ ] **Step 1: Write failing tests for exact binning and equal-loss identity**

```python
import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.phase_weighting import (
    aggregate_phase_weighted_gradients,
    phase_bin_indices,
    phase_robust_weights,
)


def test_five_bins_cover_all_499_start_phases():
    phases = jnp.array([0, 99, 100, 199, 200, 299, 300, 399, 400, 498])
    np.testing.assert_array_equal(
        phase_bin_indices(phases, phase_count=499, bin_count=5),
        np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
    )


def test_equal_bin_losses_produce_unit_detached_weights():
    losses = jnp.ones(5)
    phases = jnp.array([0, 100, 200, 300, 400])
    result = phase_robust_weights(
        losses,
        phases,
        phase_count=499,
        bin_count=5,
        robust_fraction=0.5,
    )
    np.testing.assert_array_equal(result.bin_weights, np.ones(5))
    np.testing.assert_array_equal(result.env_weights, np.ones(5))
    np.testing.assert_array_equal(
        jax.jacrev(
            lambda values: phase_robust_weights(
                values,
                phases,
                phase_count=499,
                bin_count=5,
                robust_fraction=0.5,
            ).env_weights
        )(losses),
        np.zeros((5, 5)),
    )
    gradients = {"w": jnp.arange(10, dtype=jnp.float32).reshape(5, 2)}
    weighted = aggregate_phase_weighted_gradients(
        gradients, result.env_weights
    )
    np.testing.assert_allclose(weighted["w"], gradients["w"].mean(axis=0))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_phase_weighting.py -q`

Expected: collection fails with `ModuleNotFoundError: src.algorithms.shac.phase_weighting`.

- [ ] **Step 3: Implement bin assignment and detached bounded weights**

```python
"""Detached phase-robust weighting for SHAC actor gradients."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jp


PyTree = Any


class PhaseWeighting(NamedTuple):
    bin_counts: jax.Array
    bin_losses: jax.Array
    bin_weights: jax.Array
    env_weights: jax.Array
    unweighted_loss: jax.Array
    weighted_loss: jax.Array
    valid: jax.Array


def phase_bin_indices(phases, *, phase_count: int, bin_count: int):
    if phase_count < 1 or bin_count < 1:
        raise ValueError("phase_count and bin_count must be positive")
    phases = jp.asarray(phases, dtype=jp.int32)
    return jp.minimum((phases * bin_count) // phase_count, bin_count - 1)


def phase_robust_weights(
    losses,
    phases,
    *,
    phase_count: int,
    bin_count: int,
    robust_fraction: float,
) -> PhaseWeighting:
    if not 0.0 <= robust_fraction <= 1.0:
        raise ValueError("robust_fraction must be in [0, 1]")
    losses = jp.asarray(losses)
    phases = jp.asarray(phases)
    if losses.ndim != 1 or phases.shape != losses.shape:
        raise ValueError("losses and phases must be matching vectors")

    detached_losses = jax.lax.stop_gradient(losses)
    bins = phase_bin_indices(
        phases, phase_count=phase_count, bin_count=bin_count
    )
    counts = jp.zeros((bin_count,), dtype=jp.int32).at[bins].add(1)
    occupied = counts > 0
    sums = jp.zeros((bin_count,), dtype=losses.dtype).at[bins].add(
        detached_losses
    )
    bin_losses = jp.where(occupied, sums / jp.maximum(counts, 1), jp.nan)
    all_finite = jp.all(jp.isfinite(detached_losses))

    occupied_count = jp.sum(occupied).astype(losses.dtype)
    safe_losses = jp.where(occupied & jp.isfinite(bin_losses), bin_losses, 0.0)
    mean = jp.sum(safe_losses) / jp.maximum(occupied_count, 1.0)
    variance = (
        jp.sum(jp.where(occupied, jp.square(safe_losses - mean), 0.0))
        / jp.maximum(occupied_count, 1.0)
    )
    std = jp.sqrt(variance)
    standardized = jp.where(std >= 1e-6, (safe_losses - mean) / std, 0.0)
    probabilities = jax.nn.softmax(
        jp.where(occupied, standardized, -jp.inf)
    )
    raw_bin_weights = (
        (1.0 - robust_fraction)
        + robust_fraction * occupied_count * probabilities
    )
    raw_bin_weights = jp.where(occupied, raw_bin_weights, 0.0)
    raw_bin_weights = jp.where(
        all_finite,
        raw_bin_weights,
        occupied.astype(losses.dtype),
    )
    env_weights = raw_bin_weights[bins]
    env_weights = env_weights / jp.mean(env_weights)
    env_weights = jax.lax.stop_gradient(env_weights)
    raw_bin_weights = jax.lax.stop_gradient(raw_bin_weights)
    return PhaseWeighting(
        bin_counts=counts,
        bin_losses=bin_losses,
        bin_weights=raw_bin_weights,
        env_weights=env_weights,
        unweighted_loss=jp.mean(detached_losses),
        weighted_loss=jp.sum(detached_losses * env_weights)
        / jp.sum(env_weights),
        valid=all_finite,
    )
```

- [ ] **Step 4: Run the identity tests and verify GREEN**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_phase_weighting.py -q`

Expected: the two initial tests pass.

- [ ] **Step 5: Add failing tests for hard-bin emphasis and nonfinite gradients**

```python
def test_unique_hard_bin_is_largest_bounded_and_mean_one():
    losses = jnp.array([1.0, 1.0, 1.0, 5.0, 1.0, 1.0])
    phases = jnp.array([0, 50, 100, 200, 300, 400])
    result = jax.jit(
        lambda x: phase_robust_weights(
            x,
            phases,
            phase_count=499,
            bin_count=5,
            robust_fraction=0.5,
        )
    )(losses)
    assert int(jnp.argmax(result.bin_weights)) == 3
    assert float(result.bin_weights[result.bin_counts > 0].min()) > 0.5
    assert float(result.bin_weights.max()) <= 3.0
    np.testing.assert_allclose(result.env_weights.mean(), 1.0, atol=1e-7)


def test_nonfinite_gradient_elements_have_no_numerator_or_denominator():
    gradients = {
        "w": jnp.array([[2.0, jnp.inf], [4.0, 8.0], [jnp.nan, 14.0]])
    }
    weights = jnp.array([1.0, 2.0, 3.0])
    aggregate = aggregate_phase_weighted_gradients(gradients, weights)
    np.testing.assert_allclose(aggregate["w"], np.array([10.0 / 3.0, 11.6]))


def test_no_finite_gradient_contributor_emits_zero():
    aggregate = aggregate_phase_weighted_gradients(
        {"w": jnp.array([[jnp.nan], [jnp.inf]])}, jnp.ones(2)
    )
    np.testing.assert_array_equal(aggregate["w"], np.zeros(1))
```

- [ ] **Step 6: Run the new tests and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_phase_weighting.py -q`

Expected: the aggregation tests fail because `aggregate_phase_weighted_gradients` is absent.

- [ ] **Step 7: Implement the finite-aware weighted pytree mean**

```python
def aggregate_phase_weighted_gradients(
    per_env_grads: PyTree, env_weights: jax.Array
) -> PyTree:
    leaves = jax.tree_util.tree_leaves(per_env_grads)
    env_weights = jp.asarray(env_weights)
    if not leaves or env_weights.ndim != 1:
        raise ValueError("gradient tree and vector weights are required")
    num_envs = env_weights.shape[0]
    if any(leaf.ndim < 1 or leaf.shape[0] != num_envs for leaf in leaves):
        raise ValueError("all gradient leaves must share the weight axis")

    def weighted_mean(leaf):
        shape = (num_envs,) + (1,) * (leaf.ndim - 1)
        weights = env_weights.reshape(shape)
        finite = jp.isfinite(leaf) & jp.isfinite(weights)
        numerator = jp.sum(jp.where(finite, leaf * weights, 0.0), axis=0)
        denominator = jp.sum(jp.where(finite, weights, 0.0), axis=0)
        return jp.where(denominator > 0.0, numerator / denominator, 0.0)

    return jax.tree_util.tree_map(weighted_mean, per_env_grads)
```

- [ ] **Step 8: Verify the pure module and commit it**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_phase_weighting.py tests/test_shac_gradient_aggregation.py -q`

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check src/algorithms/shac/phase_weighting.py tests/test_shac_phase_weighting.py && git diff --check`

Expected: all tests and static checks pass.

```bash
git add src/algorithms/shac/phase_weighting.py tests/test_shac_phase_weighting.py
git commit -m "feat: add phase-robust SHAC gradient weights"
```

### Task 2: Opt-in trainer integration and telemetry

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tests/test_shac_phase_weighting.py`

**Interfaces:**
- Adds `train(..., actor_phase_robust_weighting=False, actor_phase_bin_count=5, actor_phase_robust_fraction=0.5)`.
- Consumes the initial rollout phase from `updated_env_state.info["phase"]`.
- Adds phase-bin counts, losses, raw weights, validity, and weighted actor loss to diagnostic metrics only when enabled.

- [ ] **Step 1: Write failing opt-in contract tests**

```python
import inspect

from src.algorithms.shac.algorithm import train


def test_train_phase_weighting_defaults_are_disabled_and_fixed():
    parameters = inspect.signature(train).parameters
    assert parameters["actor_phase_robust_weighting"].default is False
    assert parameters["actor_phase_bin_count"].default == 5
    assert parameters["actor_phase_robust_fraction"].default == 0.5
```

- [ ] **Step 2: Run the contract test and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_phase_weighting.py -q`

Expected: FAIL because the three trainer parameters do not exist.

- [ ] **Step 3: Add validated disabled-by-default trainer settings**

Add the imports and signature fields:

```python
from src.algorithms.shac.phase_weighting import (
    aggregate_phase_weighted_gradients,
    phase_robust_weights,
)

# In train(...)
actor_phase_robust_weighting: bool = False,
actor_phase_bin_count: int = 5,
actor_phase_robust_fraction: float = 0.5,
```

Validate before constructing the environment:

```python
if not isinstance(actor_phase_robust_weighting, bool):
    raise ValueError("actor_phase_robust_weighting must be boolean")
if (
    isinstance(actor_phase_bin_count, bool)
    or not isinstance(actor_phase_bin_count, int)
    or actor_phase_bin_count < 1
):
    raise ValueError("actor_phase_bin_count must be positive")
if (
    isinstance(actor_phase_robust_fraction, bool)
    or not math.isfinite(actor_phase_robust_fraction)
    or not 0.0 <= actor_phase_robust_fraction <= 1.0
):
    raise ValueError("actor_phase_robust_fraction must be in [0, 1]")
if actor_phase_robust_weighting and gradient_accumulation_steps != 1:
    raise ValueError("phase-robust weighting requires one population shard")
if actor_phase_robust_weighting and actor_per_env_grad_clip is not None:
    raise ValueError("phase-robust weighting cannot combine with per-env clipping")
if actor_phase_robust_weighting and not env_variant.startswith("g1_tracking"):
    raise ValueError("phase-robust weighting requires G1 reference phases")
```

- [ ] **Step 4: Integrate weighting at the existing per-environment gradient seam**

Capture the initial phases before the rollout and replace only the enabled aggregation branch:

```python
actor_start_phases = jax.lax.stop_gradient(
    updated_env_state.info["phase"]
)

# After vmap(actor_grad_fn) in the single-shard path:
if actor_phase_robust_weighting:
    phase_weighting = phase_robust_weights(
        losses,
        actor_start_phases,
        phase_count=int(env.reference_transitions),
        bin_count=actor_phase_bin_count,
        robust_fraction=actor_phase_robust_fraction,
    )
    grads = aggregate_phase_weighted_gradients(
        per_env_grads, phase_weighting.env_weights
    )
    actor_grad_stats = per_env_gradient_statistics(per_env_grads)
else:
    grads, actor_grad_stats = aggregate_env_gradients(
        per_env_grads, actor_per_env_grad_clip
    )
```

Do not change the critic branch, reset code, randomization, or the established disabled branch.

- [ ] **Step 5: Add fixed-shape metrics, JSON telemetry, and hparams**

Add enabled-only metrics inside `train_step`:

```python
if actor_phase_robust_weighting:
    metrics.update(
        {
            "actor_phase_bin_counts": phase_weighting.bin_counts,
            "actor_phase_bin_losses": phase_weighting.bin_losses,
            "actor_phase_bin_weights": phase_weighting.bin_weights,
            "actor_phase_weighting_valid": phase_weighting.valid,
            "actor_loss_weighted": phase_weighting.weighted_loss,
        }
    )
```

Build the diagnostic entry before appending and add JSON-safe values only when enabled:

```python
if actor_phase_robust_weighting:
    diag_entry.update(
        {
            "actor_phase_bin_counts": np.asarray(
                metrics["actor_phase_bin_counts"]
            ).tolist(),
            "actor_phase_bin_losses": np.asarray(
                metrics["actor_phase_bin_losses"]
            ).tolist(),
            "actor_phase_bin_weights": np.asarray(
                metrics["actor_phase_bin_weights"]
            ).tolist(),
            "actor_phase_weighting_valid": bool(
                metrics["actor_phase_weighting_valid"]
            ),
            "actor_loss_unweighted": float(metrics["actor_loss"]),
            "actor_loss_weighted": float(metrics["actor_loss_weighted"]),
        }
    )
diag_log.append(diag_entry)
```

Record in `hparams.json`:

```python
"actor_phase_robust_weighting": actor_phase_robust_weighting,
"actor_phase_bin_count": actor_phase_bin_count,
"actor_phase_robust_fraction": actor_phase_robust_fraction,
```

- [ ] **Step 6: Run focused and neighboring tests**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_shac_phase_weighting.py tests/test_shac_gradient_aggregation.py tests/test_shac_microbatch.py tests/test_shac_initialization.py -q`

Expected: all selected tests pass, including the disabled defaults.

- [ ] **Step 7: Run static checks and commit the trainer seam**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check src/algorithms/shac/algorithm.py src/algorithms/shac/phase_weighting.py tests/test_shac_phase_weighting.py`

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile src/algorithms/shac/algorithm.py src/algorithms/shac/phase_weighting.py && git diff --check`

Expected: exit 0.

```bash
git add src/algorithms/shac/algorithm.py tests/test_shac_phase_weighting.py
git commit -m "feat: weight G1 actor gradients by reference phase"
```

### Task 3: Immutable E005 runner and scientific contract test

**Files:**
- Create: `tools/run_g1_phase_robust_shac.py`
- Create: `tests/test_g1_phase_robust_shac_runner.py`

**Interfaces:**
- Consumes: `build_horizon24_kwargs(profile_name, reference_path, seed) -> dict`.
- Produces: `build_phase_robust_kwargs(profile_name, reference_path, seed) -> dict`.
- Produces: a parser exposing only solver profile, reference path, seed, and output root.

- [ ] **Step 1: Write failing exact-contract and parser tests**

```python
import unittest
from pathlib import Path


class G1PhaseRobustRunnerTest(unittest.TestCase):
    def test_contract_changes_only_phase_weighting_from_e004(self):
        from tools.run_g1_horizon24_shac import build_horizon24_kwargs
        from tools.run_g1_phase_robust_shac import build_phase_robust_kwargs

        parent = build_horizon24_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )
        parent.update(
            actor_phase_robust_weighting=False,
            actor_phase_bin_count=5,
            actor_phase_robust_fraction=0.5,
        )
        candidate = build_phase_robust_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )
        self.assertEqual(set(candidate), set(parent))
        differing = {
            name for name in parent if parent[name] != candidate[name]
        }
        self.assertEqual(differing, {"actor_phase_robust_weighting"})
        self.assertTrue(candidate["actor_phase_robust_weighting"])

    def test_parser_rejects_resume_and_scientific_overrides(self):
        from tools.run_g1_phase_robust_shac import build_parser

        parser = build_parser()
        for arguments in (
            ["--resume-from", "/tmp/checkpoint.pkl"],
            ["--robust-fraction", "1.0"],
            ["--phase-bin-count", "10"],
            ["--total-steps", "8000000"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    parser.parse_args(arguments)
```

- [ ] **Step 2: Run the runner tests and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_g1_phase_robust_shac_runner.py -q`

Expected: FAIL because `tools.run_g1_phase_robust_shac` does not exist.

- [ ] **Step 3: Implement the immutable successor runner**

```python
"""Run the bounded phase-robust G1 SHAC experiment."""

import argparse
import os
from pathlib import Path

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.run_g1_horizon24_shac import build_horizon24_kwargs
from tools.run_g1_tracking_shac import configure_jax


def build_phase_robust_kwargs(
    profile_name: str, reference_path: str | Path, seed: int
) -> dict:
    kwargs = build_horizon24_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        actor_phase_robust_weighting=True,
        actor_phase_bin_count=5,
        actor_phase_robust_fraction=0.5,
    )
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Immutable G1 SHAC phase-robust early-gate run."
    )
    parser.add_argument(
        "--solver-profile",
        required=True,
        choices=tuple(sorted(SOLVER_PROFILES)),
    )
    parser.add_argument(
        "--reference-path", type=Path, default=Path(DEFAULT_REFERENCE_PATH)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("g1_phase_robust_runs")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    kwargs = build_phase_robust_kwargs(
        args.solver_profile, args.reference_path.resolve(), args.seed
    )
    profile = get_solver_profile(args.solver_profile)
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(profile):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    print((output_root / relative_save_dir).resolve())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run runner, trainer, and parent contract tests**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_g1_phase_robust_shac_runner.py tests/test_g1_horizon24_shac_runner.py tests/test_canonical_g1_shac_runner.py tests/test_shac_phase_weighting.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Run static verification and guarded argument dry run**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check tools/run_g1_phase_robust_shac.py tests/test_g1_phase_robust_shac_runner.py`

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile tools/run_g1_phase_robust_shac.py tests/test_g1_phase_robust_shac_runner.py && git diff --check`

Expected: exit 0. Do not start GPU training directly from this runner.

- [ ] **Step 6: Commit and push the verified code**

```bash
git add tools/run_g1_phase_robust_shac.py tests/test_g1_phase_robust_shac_runner.py docs/superpowers/plans/2026-08-10-g1-phase-robust-loss.md
git commit -m "feat: add bounded G1 phase-robust runner"
git push takaratruong research/g1-rmr-50hz-20260805
```

### Task 4: Register, execute, evaluate, and curate E005

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260810-005.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generated: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/experiments/E-20260810-005.md`
- Generated: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/02 Current State.md`

**Interfaces:**
- Consumes: the clean pushed code commit from Task 3 and E004's exact reference/model/solver contract.
- Produces: one guarded seed-0 run and one deterministic five-phase verdict.

- [ ] **Step 1: Register one causal change**

Create schema-v1 `E-20260810-005` with parent `E-20260810-004`, hypothesis `H-SCI-001`, seed `[0]`, GPU count `1`, wall-time budget `180` minutes, clean code commit, reference SHA-256 `bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db`, model SHA-256 `5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1`, and solver `g1-4x5-fixed-scan`.

The exact command is:

```text
/home/ubuntu/miniconda3/envs/diffsim/bin/python tools/run_g1_phase_robust_shac.py --solver-profile g1-4x5 --reference-path artifacts/E-20260808-000/reference/dance1_subject2_f122_422_50hz.npz --seed $DIFFSIM_SEED --output-root $DIFFSIM_SEED_DIR
```

Require `[policy_final.pkl, hparams.json, log.npy, diag_log.json]`. Register exactly three outcomes: `phase-robust-advances`, `phase-robust-finite-no-gain`, and `invalid-execution`. Advancement requires finite completion, complete valid phase telemetry, replay-free phase-grid minimum at least 19, and median at least 48.

- [ ] **Step 2: Validate, dry-run, approve, build, and commit the registry record**

Run: `uv run python tools/researchctl.py validate`

Run: `uv run python -m tools.runexp E-20260810-005 --dry-run`

Run: `uv run python tools/researchctl.py build-vault && uv run python tools/researchctl.py validate && git diff --check`

Expected: one seed-0 command under `runs/E-20260810-005/`, no execution during dry run, and no changes to user-owned `.obsidian` or canvas files. Commit only the experiment, current-state, and generated experiment/current-state projection.

- [ ] **Step 3: Launch through the guarded runner**

Run: `uv run python -m tools.runexp E-20260810-005`

Use the registered physical GPU only. Verify the copied contract, dirty-patch hash, first log output, process, and GPU utilization. Do not edit the pinned scientific code while the run is active.

- [ ] **Step 4: Audit completion and phase telemetry**

Require return code zero, exactly 393,216 transitions, all 247 numeric TrainState leaves finite, all four required artifacts, five-value phase-bin arrays at every diagnostic interval, finite bin losses for occupied bins, raw weights in `(0.5, 3.0]`, valid flags true, and weighted actor loss finite. Any violation selects `invalid-execution`.

- [ ] **Step 5: Evaluate the unchanged replay-free gate**

Evaluate the exact final actor under nominal physics at phases `0`, `100`, `200`, `300`, and `400` with no replay, teleportation, or post-terminal reset evidence. Select `phase-robust-advances` only if minimum survival is at least 19 and median survival is at least 48; otherwise select `phase-robust-finite-no-gain` and stop this branch.

- [ ] **Step 6: Curate and verify the result**

Run: `uv run python tools/researchctl.py validate`

Run: `uv run python tools/researchctl.py build-vault`

Run: `uv run python -m pytest tests/test_registry.py -q -k 'not links_resolve'`

Run: `git diff --check`

Update only E005, H-SCI-001 if the evaluated result creates a valid evidence edge, current state, and their generated projections. Commit and push the lab `main` branch while preserving user-owned `.obsidian` and canvas edits.
