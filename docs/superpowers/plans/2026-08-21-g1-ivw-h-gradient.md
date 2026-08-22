# G1 IVW-H Composite Gradient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the published stepwise inverse-variance composite estimator makes G1 H24 actor gradients more repeatable across action-noise tapes and more consistent across MJX solvers.

**Architecture:** Add a pure action-space IVW-H module with no environment dependency. A dedicated frozen evaluator reconstructs the exact E005 fresh/E023 populations, obtains pathwise action gradients and likelihood-ratio samples from the same rollouts, fuses them per phase/step/action coordinate, pushes each estimator through the actor, and publishes a manifest-last registered result. Training remains unchanged.

**Tech Stack:** Python 3.11, JAX x64, MuJoCo/MJX, Flax, NumPy, pytest, Ruff, existing E005 reconstruction and five-bin CAGrad reducer.

## Global Constraints

- Preserve the exact E005 checkpoint, hparams, reference, model, controller, actors, phase starts, H24 objective, action standard deviations, and two solver profiles.
- Use two fixed distinct Gaussian epsilon tapes and no other stochastic change.
- Estimate score baselines leave-one-out within each 24-replica start-phase group; do not use or change the critic.
- Fuse only at the action-mean node, per phase, step, and action dimension; do not add DDCG or full PIPPS state-distribution machinery.
- Keep the code path diagnostic-only. Do not modify SHAC `train()` or launch training in this plan.
- Stop after deterministic classification of the eight forward populations.

---

### Task 1: Pure IVW-H Action-Gradient Module

**Files:**
- Create: `src/algorithms/shac/ivw_h.py`
- Create: `tests/test_shac_ivw_h.py`

**Interfaces:**
- Consumes: reward/done arrays `(N,H)`, means/actions `(N,H,A)`, positive sigma `(A,)`, pathwise action gradients `(N,H,A)`, and phase labels `(N,)`.
- Produces: `discounted_reward_to_go`, `leave_one_out_phase_advantages`, `gaussian_mean_score_gradients`, `phase_step_action_ivw`, and `fuse_action_gradients`.

- [ ] **Step 1: Write failing return and baseline tests**

```python
def test_discounted_reward_to_go_stops_at_done():
    reward = jnp.asarray([[1.0, 2.0, 4.0, 8.0]])
    done = jnp.asarray([[False, True, False, False]])
    actual = discounted_reward_to_go(reward, done, gamma=0.5)
    np.testing.assert_allclose(actual, [[2.0, 2.0, 8.0, 8.0]])


def test_leave_one_out_baseline_excludes_own_return():
    returns = jnp.asarray([[1.0], [3.0], [9.0], [11.0]])
    phases = jnp.asarray([0, 0, 1, 1])
    actual = leave_one_out_phase_advantages(returns, phases)
    np.testing.assert_allclose(actual[:, 0], [-2.0, 2.0, -2.0, 2.0])
```

- [ ] **Step 2: Run RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_ivw_h.py
```

Expected: collection fails because `src.algorithms.shac.ivw_h` does not exist.

- [ ] **Step 3: Implement exact episodic returns and leave-one-out advantages**

```python
def discounted_reward_to_go(reward, done, *, gamma):
    reward = jp.asarray(reward)
    done = jp.asarray(done, dtype=jp.bool_)
    if reward.ndim != 2 or done.shape != reward.shape:
        raise ValueError("reward and done must have shape (N, H)")
    if not isinstance(gamma, (int, float)) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    def backward(carry, values):
        r, terminal = values
        value = r + gamma * jp.where(terminal, 0.0, carry)
        return value, value

    _, reversed_values = jax.lax.scan(
        backward,
        jp.zeros(reward.shape[0], dtype=reward.dtype),
        (reward.T[::-1], done.T[::-1]),
    )
    return reversed_values[::-1].T


def leave_one_out_phase_advantages(returns, phases):
    returns = jp.asarray(returns)
    phases = jp.asarray(phases)
    if returns.ndim != 2 or phases.shape != returns.shape[:1]:
        raise ValueError("returns/phases have incompatible shapes")
    same = phases[:, None] == phases[None, :]
    peers = same.astype(returns.dtype) - jp.eye(returns.shape[0], dtype=returns.dtype)
    counts = jp.sum(peers, axis=1)
    if bool(jp.any(counts < 1)):
        raise ValueError("every phase requires at least two replicas")
    baseline = jp.einsum("ij,jt->it", peers, returns) / counts[:, None]
    return returns - baseline
```

- [ ] **Step 4: Verify GREEN for return/baseline tests**

Run the two tests directly and expect both to pass.

- [ ] **Step 5: Write failing score, weight, and fusion tests**

```python
def test_gaussian_score_gradient_has_policy_loss_sign():
    means = jnp.asarray([[[0.0, 1.0]]])
    actions = jnp.asarray([[[2.0, 0.0]]])
    advantage = jnp.asarray([[4.0]])
    actual = gaussian_mean_score_gradients(
        means, actions, advantage, jnp.asarray([2.0, 1.0]), horizon=2
    )
    np.testing.assert_allclose(actual, [[[-1.0, 2.0]]])


def test_ivw_uses_phase_local_sample_variance():
    g0 = jnp.asarray([[[0.0]], [[2.0]], [[10.0]], [[10.0]]])
    g1 = jnp.asarray([[[0.0]], [[0.0]], [[8.0]], [[12.0]]])
    alpha = phase_step_action_ivw(g0, g1, jnp.asarray([0, 0, 1, 1]))
    np.testing.assert_allclose(alpha[:2], 1.0)
    np.testing.assert_allclose(alpha[2:], 0.0)


def test_fusion_selects_the_registered_weight_per_sample():
    actual = fuse_action_gradients(
        jnp.asarray([[[2.0]]]),
        jnp.asarray([[[10.0]]]),
        jnp.asarray([[[0.25]]]),
    )
    np.testing.assert_allclose(actual, [[[8.0]]])
```

- [ ] **Step 6: Run RED for missing estimator functions**

Run the three tests directly. Expected: import failures for the missing APIs.

- [ ] **Step 7: Implement score samples, phase-local IVW, and fusion**

```python
def gaussian_mean_score_gradients(means, sampled_actions, advantages, sigma, *, horizon):
    means = jp.asarray(means)
    actions = jp.asarray(sampled_actions)
    advantages = jp.asarray(advantages)
    sigma = jp.asarray(sigma)
    if means.ndim != 3 or actions.shape != means.shape:
        raise ValueError("means/actions must have shape (N, H, A)")
    if advantages.shape != means.shape[:2] or sigma.shape != means.shape[2:]:
        raise ValueError("advantage/sigma shapes are incompatible")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be positive")
    if not bool(jp.all(jp.isfinite(sigma) & (sigma > 0.0))):
        raise ValueError("sigma must be finite and positive")
    return (
        advantages[..., None]
        / horizon
        * (means - jax.lax.stop_gradient(actions))
        / jp.square(sigma)
    )


def phase_step_action_ivw(score_grad, pathwise_grad, phases):
    score_grad = jp.asarray(score_grad)
    pathwise_grad = jp.asarray(pathwise_grad)
    phases = jp.asarray(phases)
    if score_grad.ndim != 3 or pathwise_grad.shape != score_grad.shape:
        raise ValueError("gradient samples must share shape (N, H, A)")
    if phases.shape != score_grad.shape[:1]:
        raise ValueError("phases must have shape (N,)")
    same = phases[:, None] == phases[None, :]
    counts = jp.sum(same, axis=1)
    if bool(jp.any(counts < 2)):
        raise ValueError("every phase requires at least two replicas")
    mean0 = jp.einsum("ij,jta->ita", same, score_grad) / counts[:, None, None]
    mean1 = jp.einsum("ij,jta->ita", same, pathwise_grad) / counts[:, None, None]
    denominator = counts - 1
    var0 = jp.einsum("ij,ijta->ita", same, jp.square(score_grad[None] - mean0[:, None])) / denominator[:, None, None]
    var1 = jp.einsum("ij,ijta->ita", same, jp.square(pathwise_grad[None] - mean1[:, None])) / denominator[:, None, None]
    total = var0 + var1
    return jp.where(total > 0.0, var0 / total, 0.0)


def fuse_action_gradients(pathwise_grad, score_grad, alpha):
    if pathwise_grad.shape != score_grad.shape or alpha.shape != pathwise_grad.shape:
        raise ValueError("fusion inputs must share shape")
    return alpha * pathwise_grad + (1.0 - alpha) * score_grad
```

- [ ] **Step 8: Add nonfinite and repeated-phase validation tests**

```python
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_score_gradient_rejects_nonfinite_inputs(bad):
    means = np.zeros((2, 1, 1))
    means[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        gaussian_mean_score_gradients(
            means, np.zeros_like(means), np.ones((2, 1)), np.ones(1), horizon=1
        )


def test_ivw_rejects_singleton_phase_group():
    gradient = np.zeros((3, 1, 1))
    with pytest.raises(ValueError, match="at least two"):
        phase_step_action_ivw(gradient, gradient, np.asarray([0, 0, 1]))


def test_ivw_exact_zero_variance_selects_score_weight():
    gradient = np.zeros((2, 1, 1))
    np.testing.assert_array_equal(
        phase_step_action_ivw(gradient, gradient, np.asarray([0, 0])),
        gradient,
    )
```

Implement host-side finite and replica-count validation before converting the
inputs to JAX arrays. Run the complete focused suite and expect all tests to
pass.

- [ ] **Step 9: Static verification and commit**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check src/algorithms/shac/ivw_h.py tests/test_shac_ivw_h.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile src/algorithms/shac/ivw_h.py tests/test_shac_ivw_h.py
git diff --check
git add src/algorithms/shac/ivw_h.py tests/test_shac_ivw_h.py
git commit -m "feat: add phase-local IVW-H action gradients"
```

---

### Task 2: Frozen Paired IVW-H Gradient Evaluator

**Files:**
- Create: `tools/evaluate_g1_ivw_h_gradients.py`
- Create: `tests/test_g1_ivw_h_gradient_evaluator.py`

**Interfaces:**
- Consumes: exact E005 checkpoint/hparams/reference/runtime assets, fresh actor constructor, two solver profiles, Task 1 estimator functions, and existing five-bin `aggregate_audit_direction`.
- Produces: `build_fixed_phase_population`, `action_node_pathwise_gradients`, `push_action_gradients_to_policy`, `classify_ivw_h_gradient_audit`, `run_gradient_capture`, and manifest-last evidence.

- [ ] **Step 1: Write RED contract and classification tests**

```python
def test_registered_population_has_two_distinct_replayable_tapes():
    population = build_fixed_phase_population(seed=0)
    assert population["phase"].shape == (120,)
    assert population["noise"].shape == (2, 120, 24, 29)
    assert not np.array_equal(population["noise"][0], population["noise"][1])
    np.testing.assert_array_equal(population["phase"], np.repeat([0, 25, 50, 75, 100], 24))


def test_destructive_precedes_robust_when_retained_signal_fails():
    evidence = valid_evidence(
        solver_gain=0.08,
        tape_gain=0.08,
        retained_cosine=0.49,
    )
    assert classify_ivw_h_gradient_audit(evidence) == "ivw-h-destructive"
```

- [ ] **Step 2: Run RED**

Expected: collection fails because `tools.evaluate_g1_ivw_h_gradients` does not exist.

- [ ] **Step 3: Implement population, vector math, classification, and atomic writers**

Copy the immutable E005 constants `POPULATION=120`, phases
`(0,25,50,75,100)`, `REPLICAS_PER_PHASE=24`, `HORIZON=24`, `ACTION_DIM=29`,
the two solver names, and all five input/runtime hashes into the evaluator.
Derive tape zero with NumPy seed `913024` to preserve E005 and tape one with
fixed seed `913025`:

```python
def build_fixed_phase_population(seed):
    if seed != 0:
        raise ValueError("IVW-H gradient seed must be zero")
    phase = np.repeat(PHASES, REPLICAS_PER_PHASE)
    noise = np.stack(
        [
            np.random.default_rng(tape_seed).standard_normal(
                (POPULATION, HORIZON, ACTION_DIM), dtype=np.float32
            )
            for tape_seed in (913_024, 913_025)
        ]
    )
    return {"phase": phase, "noise": noise}
```

Implement classification directly from the design thresholds. Validate both
actor rows, all scalar and five-element phase values, finite counts, parity,
and nonzero vectors before classification. Use the existing E005 `_atomic_json`,
`_atomic_npz`, `_vector_cosine`, `_tree_vector`, `_tree_matrix`, clean-source
validation, and completion-reopen implementations without changing their
semantics; import them rather than duplicating them.

- [ ] **Step 4: Verify pure evaluator GREEN**

Run population, comparison, classification, artifact hash, and tamper tests only.

- [ ] **Step 5: Write RED action-node VJP parity test**

```python
def test_action_node_vjp_matches_direct_recurrent_pathwise_gradient():
    direct = direct_parameter_gradient(synthetic_actor, params, initial, epsilon)
    action_grad, observations = action_node_pathwise_gradients(
        synthetic_actor, params, initial, epsilon
    )
    pushed = push_action_gradients_to_policy(
        synthetic_actor, params, observations, action_grad
    )
    np.testing.assert_allclose(tree_vector(pushed), tree_vector(direct), rtol=1e-10, atol=1e-10)
```

- [ ] **Step 6: Run RED and implement action-node capture/VJP**

The rollout accepts an `(H,A)` additive `delta` after reparameterized noise and before `env.step`. Differentiate the unchanged negative-return objective with respect to `delta` at exact zeros. Cache normalized actor observations and means. For each stopped action-gradient tensor, differentiate the scalar surrogate

```python
jp.sum(stop(action_gradient) * actor.apply(parameters, stop(observations)))
```

with respect to parameters. Reject any mismatch with the direct actor gradient before scientific aggregation.

- [ ] **Step 7: Implement the exact frozen capture**

Reconstruct initial states once per solver and require bit identity. For every actor/solver/tape, run the ordinary rollout, compute direct and action-node pathwise gradients, reward-to-go, leave-one-out score samples, IVW-H weights, and policy gradients for all three estimators. Process 24-environment phase chunks to bound memory. Require at least 16 finite parameter-gradient contributors per phase and aggregate with unchanged norm-1 five-bin CAGrad.

- [ ] **Step 8: Persist and validate complete evidence**

Write complete action-space arrays plus aggregate/phase parameter vectors to `gradient_evidence.npz`; write all solver/tape/phase cosines, retained cosines/norms, alpha distributions, and finite counts to `gradient_summary.json`; plot ordinary versus IVW-H solver/tape cosines; then write `completion.json` last. Reopen and hash-check every artifact.

- [ ] **Step 9: Run the compiled smoke**

```bash
CUDA_VISIBLE_DEVICES=2 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m tools.evaluate_g1_ivw_h_gradients \
  --checkpoint /home/ubuntu/projects/diffsim2real-lab/runs/E-20260814-023/20260815T023359Z/seed-0/training_runs/training_runs/shac_20260814_193402/checkpoint_step_1572864.pkl \
  --hparams /home/ubuntu/projects/diffsim2real-lab/runs/E-20260814-023/20260815T023359Z/seed-0/training_runs/training_runs/shac_20260814_193402/hparams.json \
  --reference-path /home/ubuntu/projects/diffsim2real/outputs/rmr_motion_walk_win137_212_named.npz \
  --repository /home/ubuntu/worktrees/open-diffloco/e041-evaluator-20260815 \
  --code-commit "$(git rev-parse HEAD)" \
  --output-directory /tmp/g1-ivw-h-smoke \
  --seed 0 --smoke --smoke-phase 25 --smoke-replicas 8
```

Expected: valid non-scientific smoke, finite nonzero pathwise/score/fused action gradients, and direct-VJP cosine at least `0.999`.

- [ ] **Step 10: Focused verification and commit**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_shac_ivw_h.py tests/test_g1_ivw_h_gradient_evaluator.py \
  tests/test_g1_contact_topology_gradient_evaluator.py \
  tests/test_shac_cagrad.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
  src/algorithms/shac/ivw_h.py tools/evaluate_g1_ivw_h_gradients.py \
  tests/test_shac_ivw_h.py tests/test_g1_ivw_h_gradient_evaluator.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
  src/algorithms/shac/ivw_h.py tools/evaluate_g1_ivw_h_gradients.py
git diff --check
git add tools/evaluate_g1_ivw_h_gradients.py tests/test_g1_ivw_h_gradient_evaluator.py
git commit -m "feat: evaluate G1 IVW-H gradient reliability"
```

---

### Task 3: Preregister, Execute, And Curate The Bounded Experiment

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260821-006.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Create or modify on decisive evidence: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/findings/F-20260821-002 Composite Gradient Reliability.md`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/runtime/vault-previews.yaml`

**Interfaces:**
- Consumes: clean Task 2 code commit, exact E005 parent inputs, registered two-tape frozen protocol.
- Produces: one evaluated experiment, immutable run artifacts, current decision, and regenerated Dashboard projection.

- [ ] **Step 1: Write one experiment record**

Pin the exact clean code SHA, empty dirty patch hash, conda environment, E005 input/runtime hashes, solver pair, seed zero, GPU 2, 120-minute budget, evaluator command, outcome map, and required artifacts. Parent E005 and change only the estimator.

- [ ] **Step 2: Validate and dry-run**

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260821-006 --dry-run
```

Expected: registry valid and command/environment/provenance render exactly once.

- [ ] **Step 3: Approve and execute**

Set `approved_by: codex`, a UTC timestamp, then run:

```bash
uv run python -m tools.runexp E-20260821-006
```

Do not launch another experiment while it runs.

- [ ] **Step 4: Curate only deterministic evidence**

Reopen `completion.json`, verify every artifact hash, recompute the registered classifier from `gradient_summary.json`, and update the experiment result/current state with exactly one registered outcome. Promote a finding only if the result establishes a reusable causal boundary under the skill's promotion rule.

- [ ] **Step 5: Rebuild and verify Obsidian**

```bash
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py build-vault
uv run python -m pytest -q tests/test_registry.py -k 'dashboard or vault'
git diff --check
```

Copy the canonical cosine plot to the ignored/untracked preview attachment path via the existing manifest/build flow. Do not hand-edit the Dashboard.
