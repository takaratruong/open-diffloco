# Random RMR Capacity and Gradient Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a one-GPU preflight for a randomly initialized exact-RMR G1 actor/critic, determine the largest useful SHAC batch that fits, and compare matched SHAC and PPO-style gradient variance/directions without loading pretrained policy data.

**Architecture:** Add a pure-JAX RMR training policy separate from the frozen-checkpoint `RmrPolicy`, a pure G1 training-data distribution for perturbed reference-state initialization and observation corruption, and a dedicated one-update probe rather than adding more branches to the Go2 trainer. The probe differentiates a batch-mean loss directly, writes strict finite JSON, and exposes the same rollout to a pathwise SHAC estimator and a stopped-simulator PPO-style estimator.

**Tech Stack:** Python 3.10, JAX/Optax/Flax, MuJoCo MJX 3.9, NumPy, `unittest`/pytest, existing G1 RMR environment and fixed-solver scope, DiffSim2Real research registry.

## Global Constraints

- Actor architecture: `154 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 29`, ELU hidden activations, linear output, no LayerNorm.
- Critic architecture: `286 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 1`, ELU hidden activations, linear output, no LayerNorm.
- Every linear parameter uses `Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`; actor `log_std` starts at zero.
- No PPO checkpoint, source-policy action, action tape, behavior-cloning loss, residual controller, or pretrained normalization statistic may be read.
- Actor and critic normalizers start at zero mean/unit variance with epsilon `1e-2`.
- Retain validated source-order RMR observations/actions, 50 Hz timebase, solver `4/5`, nominal model, RMR rewards/terminations, and unbounded action support.
- The initial differentiable horizon is 12.
- Capacity tiers are ordered `256`, `512`, `1024`, `2048` on one L40S; stop after the first failed tier.
- Compilation must not advance logical PRNG, environment, normalizer, sampler, actor, critic, or optimizer state.
- All GPU runs must be preregistered; no locomotion claim may come from the capacity probe.
- Use TDD: observe every focused test fail for the intended missing behavior before production edits.
- Preserve existing Go2, generic G1 SHAC, residual, and pretrained full-policy behavior.

---

## File Map

- Create `src/core/rmr_training_policy.py`: random RMR MLP parameters, initialization, deterministic application, Gaussian action sampling, entropy, and parameter facts.
- Create `tests/test_rmr_training_policy.py`: exact architecture/initialization/application/gradient tests.
- Create `src/envs/g1_tracking/training_distribution.py`: adaptive phase sampler, perturbed RSI reset, and actor-observation corruption.
- Create `tests/test_g1_tracking_training_distribution.py`: deterministic range, sampler, and corruption-boundary tests.
- Create `src/algorithms/shac/batch_gradients.py`: direct batch-mean differentiation, tree geometry, and online variance summaries.
- Create `tests/test_shac_batch_gradients.py`: explicit-reference tests for mean gradients and moments.
- Create `src/algorithms/shac/g1_rmr_probe.py`: exact one-update G1 rollout, SHAC actor loss, critic boundary, and PPO-style matched estimator.
- Create `tests/test_g1_rmr_probe.py`: construction, state-purity, stopped-gradient, finite-result, and artifact tests.
- Create `tools/probe_g1_rmr_capacity.py`: isolated tier worker and parent capacity ladder with atomic strict JSON.
- Create `tools/compare_g1_rmr_gradients.py`: eight-batch matched SHAC/PPO diagnostic and strict JSON.
- Modify `tools/evaluate_g1_tracking.py`: load the new deterministic random-RMR checkpoint format without source inputs.
- Modify `tests/test_g1_tracking_evaluator.py`: checkpoint-loading and exclusivity tests.
- Create research experiment and hypothesis YAML only after code verification in the DiffSim2Real-Lab repository.

---

### Task 1: Exact Random RMR Training Policy

**Files:**
- Create: `src/core/rmr_training_policy.py`
- Create: `tests/test_rmr_training_policy.py`

**Interfaces:**
- Consumes: JAX PRNG keys and normalized float32 observations.
- Produces:
  - `RmrMlpParams(weights: tuple[jax.Array, ...], biases: tuple[jax.Array, ...])`
  - `GaussianRmrActorParams(mlp: RmrMlpParams, log_std: jax.Array)`
  - `init_rmr_mlp(key, input_dim, hidden_dims, output_dim, dtype)`
  - `init_gaussian_rmr_actor(key, input_dim=154, action_dim=29, dtype=jnp.float32)`
  - `init_rmr_critic(key, input_dim=286, dtype=jnp.float32)`
  - `apply_rmr_mlp(params, observations)`
  - `sample_rmr_action(params, normalized_observations, epsilon)`
  - `gaussian_entropy(log_std)`
  - `rmr_mlp_parameter_count(params)`

- [ ] **Step 1: Write failing architecture and initialization tests**

```python
def test_exact_actor_architecture_and_nonzero_output_head():
    params = init_gaussian_rmr_actor(jax.random.PRNGKey(42))
    assert tuple(w.shape for w in params.mlp.weights) == (
        (2048, 154), (2048, 2048), (1024, 2048), (1024, 1024),
        (512, 1024), (512, 512), (29, 512),
    )
    assert rmr_mlp_parameter_count(params.mlp) == 8_463_901
    assert np.any(np.asarray(params.mlp.weights[-1]) != 0.0)
    np.testing.assert_array_equal(np.asarray(params.log_std), np.zeros(29))


def test_every_linear_parameter_respects_pytorch_default_bounds():
    params = init_gaussian_rmr_actor(jax.random.PRNGKey(7))
    for weight, bias in zip(params.mlp.weights, params.mlp.biases):
        bound = 1.0 / np.sqrt(weight.shape[1])
        assert np.max(np.abs(np.asarray(weight))) <= bound
        assert np.max(np.abs(np.asarray(bias))) <= bound
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest \
  tests/test_rmr_training_policy.py -q
```

Expected: collection fails because `src.core.rmr_training_policy` does not exist.

- [ ] **Step 3: Implement immutable parameters and distribution-equivalent initialization**

```python
class RmrMlpParams(NamedTuple):
    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]


class GaussianRmrActorParams(NamedTuple):
    mlp: RmrMlpParams
    log_std: jax.Array


RMR_HIDDEN_DIMS = (2048, 2048, 1024, 1024, 512, 512)


def init_rmr_mlp(key, input_dim, hidden_dims, output_dim, dtype=jnp.float32):
    dims = (input_dim, *hidden_dims, output_dim)
    keys = jax.random.split(key, 2 * (len(dims) - 1))
    weights, biases = [], []
    for index, (fan_in, fan_out) in enumerate(zip(dims, dims[1:])):
        bound = jnp.asarray(1.0 / math.sqrt(fan_in), dtype=dtype)
        weights.append(jax.random.uniform(
            keys[2 * index], (fan_out, fan_in), dtype=dtype,
            minval=-bound, maxval=bound,
        ))
        biases.append(jax.random.uniform(
            keys[2 * index + 1], (fan_out,), dtype=dtype,
            minval=-bound, maxval=bound,
        ))
    return RmrMlpParams(tuple(weights), tuple(biases))
```

Implement actor/critic constructors by splitting the caller key once and using
the frozen dimensions above. Do not import PyTorch.

- [ ] **Step 4: Add failing application, entropy, and full-gradient tests**

```python
def test_sampling_uses_caller_owned_epsilon_exactly():
    params = init_gaussian_rmr_actor(jax.random.PRNGKey(1))
    obs = jnp.zeros((3, 154), dtype=jnp.float32)
    epsilon = jnp.arange(87, dtype=jnp.float32).reshape(3, 29) / 100.0
    mean = apply_rmr_mlp(params.mlp, obs)
    np.testing.assert_allclose(
        sample_rmr_action(params, obs, epsilon),
        mean + epsilon,
        rtol=0.0,
        atol=1e-7,
    )


def test_every_actor_layer_and_log_std_receives_gradient():
    params = init_gaussian_rmr_actor(jax.random.PRNGKey(2))
    obs = jnp.ones((2, 154), dtype=jnp.float32)
    epsilon = jnp.ones((2, 29), dtype=jnp.float32)
    grads = jax.grad(
        lambda value: jnp.sum(sample_rmr_action(value, obs, epsilon))
    )(params)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(leaf)).all()
        assert np.linalg.norm(np.asarray(leaf)) > 0.0
```

- [ ] **Step 5: Implement ELU application, sampling, entropy, and counts**

```python
def apply_rmr_mlp(params, observations):
    value = jnp.asarray(observations, dtype=params.weights[0].dtype)
    last = len(params.weights) - 1
    for index, (weight, bias) in enumerate(zip(params.weights, params.biases)):
        value = jnp.matmul(value, weight.T, precision=lax.Precision.HIGHEST) + bias
        if index != last:
            value = jax.nn.elu(value)
    return value


def sample_rmr_action(params, normalized_observations, epsilon):
    mean = apply_rmr_mlp(params.mlp, normalized_observations)
    return mean + jnp.exp(params.log_std) * jnp.asarray(epsilon, mean.dtype)


def gaussian_entropy(log_std):
    constant = 0.5 * (1.0 + math.log(2.0 * math.pi))
    return jnp.sum(log_std + constant)
```

- [ ] **Step 6: Run focused and adjacent tests**

Run:

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest \
  tests/test_rmr_training_policy.py tests/test_rmr_policy.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/core/rmr_training_policy.py tests/test_rmr_training_policy.py
git commit -m "feat: add random RMR training policy"
```

---

### Task 2: RMR Training Reset, Phase Sampling, and Observation Noise

**Files:**
- Create: `src/envs/g1_tracking/training_distribution.py`
- Create: `tests/test_g1_tracking_training_distribution.py`

**Interfaces:**
- Consumes: `G1TrackingEnv`, caller PRNG keys, phase/failure arrays.
- Produces:
  - `PhaseSamplerState(failed_count: jax.Array)`
  - `init_phase_sampler(reference_length, control_hz=50)`
  - `phase_sampling_probabilities(state, uniform_ratio=0.2)`
  - `update_phase_sampler(state, phases, terminals, reference_length, alpha=0.001)`
  - `sample_training_phase(key, state, reference_length)`
  - `reset_training_at_phase(env, key, difficulty, phase)`
  - `corrupt_actor_observation(key, observation)`

- [ ] **Step 1: Write failing phase-sampler tests**

```python
def test_phase_sampler_starts_uniform_and_moves_toward_failed_bin():
    state = init_phase_sampler(reference_length=212)
    before = np.asarray(phase_sampling_probabilities(state))
    np.testing.assert_allclose(before, np.full_like(before, 1.0 / len(before)))
    phases = jnp.array([10, 10, 10, 170], dtype=jnp.int32)
    terminals = jnp.array([1, 1, 1, 0], dtype=jnp.float32)
    updated = update_phase_sampler(state, phases, terminals, 212)
    after = np.asarray(phase_sampling_probabilities(updated))
    assert after[0] > before[0]
```

- [ ] **Step 2: Run the focused test and verify RED**

Expected: import failure for the missing module.

- [ ] **Step 3: Implement sampler state and pure updates**

```python
class PhaseSamplerState(NamedTuple):
    failed_count: jax.Array


def _bin_count(reference_length, control_hz=50):
    return int(reference_length // control_hz) + 1


def phase_sampling_probabilities(state, uniform_ratio=0.2):
    count = state.failed_count + uniform_ratio / state.failed_count.shape[0]
    return count / jnp.sum(count)


def update_phase_sampler(state, phases, terminals, reference_length, alpha=0.001):
    bins = jnp.minimum(
        phases * state.failed_count.shape[0] // reference_length,
        state.failed_count.shape[0] - 1,
    )
    failures = jnp.bincount(
        bins, weights=terminals, length=state.failed_count.shape[0]
    )
    return PhaseSamplerState(
        (1.0 - alpha) * state.failed_count + alpha * failures
    )
```

- [ ] **Step 4: Write failing perturbed-RSI and observation-slice tests**

Use the real validated environment. Assert caller-selected phase preservation,
all root/joint perturbation bounds, model-order joint clipping, and these exact
actor observation slices:

```python
ACTOR_NOISE_SLICES = (
    (slice(58, 64), 0.05),
    (slice(64, 67), 0.2),
    (slice(67, 96), 0.01),
    (slice(96, 125), 0.5),
)


def test_unregistered_actor_fields_are_byte_exact():
    obs = jnp.arange(154, dtype=jnp.float64)
    noisy = np.asarray(corrupt_actor_observation(jax.random.PRNGKey(4), obs))
    registered = np.zeros(154, dtype=bool)
    for region, _ in ACTOR_NOISE_SLICES:
        registered[region] = True
    np.testing.assert_array_equal(noisy[~registered], np.asarray(obs)[~registered])
```

- [ ] **Step 5: Implement perturbed reset and corruption**

Split the reset key into phase-independent position, orientation, velocity,
joint, and continuation keys. Start from `env.qpos_reference[phase]` and
`env.qvel_reference[phase]`; apply the exact ranges from the design; call
`mjx.forward`; then construct the same `EnvState` fields as
`G1TrackingEnv.reset_at_phase`. Do not modify exact evaluation reset behavior.

- [ ] **Step 6: Run focused and adjacent environment tests**

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest \
  tests/test_g1_tracking_training_distribution.py \
  tests/test_g1_tracking_environment.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/envs/g1_tracking/training_distribution.py \
  tests/test_g1_tracking_training_distribution.py
git commit -m "feat: add RMR G1 training distribution"
```

---

### Task 3: Direct Batch Gradients and Online Gradient Geometry

**Files:**
- Create: `src/algorithms/shac/batch_gradients.py`
- Create: `tests/test_shac_batch_gradients.py`

**Interfaces:**
- Produces:
  - `batch_mean_value_and_grad(single_loss, params, *batched_inputs)`
  - `tree_dot(left, right)`, `tree_norm(tree)`, `tree_cosine(left, right)`
  - `TreeMoments(count, mean, mean_squared_norm)`
  - `init_tree_moments(tree)`, `update_tree_moments(state, tree)`
  - `finalize_tree_moments(state)` returning mean norm, trace variance, and SNR.

- [ ] **Step 1: Write a failing equivalence test**

```python
def test_batch_mean_gradient_equals_explicit_per_example_mean():
    params = {"w": jnp.array([0.5, -1.0])}
    x = jnp.array([[1.0, 2.0], [-2.0, 3.0], [4.0, -1.0]])

    def loss_one(value, sample):
        return jnp.square(jnp.dot(value["w"], sample))

    _, direct = batch_mean_value_and_grad(loss_one, params, x)
    explicit = jax.tree_util.tree_map(
        lambda leaf: jnp.mean(leaf, axis=0),
        jax.vmap(jax.grad(loss_one), in_axes=(None, 0))(params, x),
    )
    np.testing.assert_allclose(direct["w"], explicit["w"], atol=1e-7)
```

- [ ] **Step 2: Verify RED, then implement direct mean differentiation**

```python
def batch_mean_value_and_grad(single_loss, params, *batched_inputs):
    def mean_loss(value):
        losses = jax.vmap(single_loss, in_axes=(None,) + (0,) * len(batched_inputs))(
            value, *batched_inputs
        )
        return jnp.mean(losses)
    return jax.value_and_grad(mean_loss)(params)
```

The production probe may use an equivalent batched rollout loss rather than
this convenience wrapper, but its test must pass through this public boundary.

- [ ] **Step 3: Write failing tree-geometry and Welford-moment tests**

Use three small gradient trees and compare online results with explicitly
stacked flattened vectors. Require finite zero-variance handling: identical
gradients yield trace variance zero and SNR `inf`, while a zero mean with
nonzero variance yields SNR zero.

- [ ] **Step 4: Implement tree geometry and online moments**

Accumulate the mean tree with Welford updates and the scalar sum of squared
deviations; never concatenate production 8.46M-parameter gradients.

- [ ] **Step 5: Run tests and commit**

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest \
  tests/test_shac_batch_gradients.py tests/test_shac_gradient_aggregation.py -q
git add src/algorithms/shac/batch_gradients.py tests/test_shac_batch_gradients.py
git commit -m "feat: add batch gradient diagnostics"
```

---

### Task 4: Dedicated One-Update G1 RMR Probe

**Files:**
- Create: `src/algorithms/shac/g1_rmr_probe.py`
- Create: `tests/test_g1_rmr_probe.py`

**Interfaces:**
- Consumes Tasks 1-3 and `G1TrackingRMR50HzValidatedEnv`.
- Produces:
  - `G1RmrProbeConfig(num_envs, horizon=12, seed=42, actor_lr=1e-3, critic_lr=5e-4, critic_iterations=16)`
  - `G1RmrProbeState`
  - `initialize_probe(env, config)`
  - `build_probe_step(env, config)`
  - `run_probe_tier(env, config)` returning a strict finite Python mapping.

- [ ] **Step 1: Write failing initialization and compilation-purity tests**

```python
def test_probe_initialization_has_no_pretrained_input_and_exact_shapes():
    config = G1RmrProbeConfig(num_envs=2, horizon=1, seed=42)
    state = initialize_probe(_validated_env(), config)
    assert rmr_mlp_parameter_count(state.actor_params.mlp) == 8_463_901
    assert rmr_mlp_parameter_count(state.critic_params) == 8_719_873
    assert float(state.actor_normalizer.count) == pytest.approx(1e-4)


def test_lowering_probe_step_does_not_advance_state():
    env = _validated_env()
    config = G1RmrProbeConfig(num_envs=1, horizon=1, seed=42)
    state = initialize_probe(env, config)
    before = jax.tree_util.tree_map(np.asarray, state)
    build_probe_step(env, config).lower(state).compile()
    after = jax.tree_util.tree_map(np.asarray, state)
    for left, right in zip(jax.tree_util.tree_leaves(before),
                           jax.tree_util.tree_leaves(after)):
        np.testing.assert_array_equal(left, right)
```

- [ ] **Step 2: Verify RED**

Expected: missing `g1_rmr_probe` module.

- [ ] **Step 3: Implement probe configuration and initialization**

Use `Normalizer(154, eps=1e-2)` and `Normalizer(286, eps=1e-2)`, Task-1 actor
and critic, Task-2 adaptive sampler/reset, and Optax Adam with aggregate global
norm clipping `1.0`. Keep model parameters float32 and simulation state float64.

- [ ] **Step 4: Write a failing finite one-step test at `num_envs=1`, horizon 1**

The result must include exact actor/critic parameter counts, initial/final tree
finiteness, actor and critic gradient norms, mean reward, environment count,
horizon, logical steps, and a proof that actor parameters changed while the
input state remained unchanged.

- [ ] **Step 5: Implement the batched SHAC rollout and critic boundary**

Implement these exact data-flow rules:

```python
normalized = env.normalize_actor_obs(
    actor_normalizer, actor_norm_state, corrupt_actor_observation(obs_key, state.obs)
).astype(jnp.float32)
action = sample_rmr_action(actor_params, normalized, epsilon).astype(jnp.float64)
next_state = env.step(state, action)
```

The actor loss is the negative mean 12-step return divided by horizon, with
target-critic bootstrap and `-0.005 * gaussian_entropy(log_std)`. Differentiate
the batch mean once; do not vmap `jax.grad`. The critic uses the existing
TD(lambda) equations with 16 aggregate batch-gradient updates. Update
normalizers only in the returned logical state.

- [ ] **Step 6: Compile without warm-up mutation and emit a finite tier result**

Use:

```python
compiled = jax.jit(step).lower(state).compile()
new_state, metrics = compiled(state)
jax.block_until_ready(new_state.step)
```

Do not execute an unrecorded warm-up step or transplant warm-up normalizers.

- [ ] **Step 7: Run focused tests**

```bash
CUDA_VISIBLE_DEVICES=7 JAX_ENABLE_X64=true \
  /home/ubuntu/miniconda3/envs/rl/bin/python -m pytest \
  tests/test_g1_rmr_probe.py -q
```

Expected: all CPU/pure tests pass; mark the real MJX compilation test with the
existing opt-in GPU convention if it exceeds the normal focused-test budget.

- [ ] **Step 8: Commit**

```bash
git add src/algorithms/shac/g1_rmr_probe.py tests/test_g1_rmr_probe.py
git commit -m "feat: add random RMR G1 SHAC probe"
```

---

### Task 5: Isolated One-GPU Capacity Ladder

**Files:**
- Create: `tools/probe_g1_rmr_capacity.py`
- Modify: `tests/test_g1_rmr_probe.py`

**Interfaces:**
- Parent CLI:
  - `--environment-counts 256 512 1024 2048`
  - `--horizon 12`
  - `--seed 42`
  - `--output PATH`
  - `--tier-timeout-minutes N`
- Worker CLI (private): `--worker-environments N --worker-output PATH`.

- [ ] **Step 1: Write failing CLI and outcome-classification tests**

Use a fake worker command to prove:

- tiers run in order;
- parent stops after the first nonzero, timeout, nonfinite, or missing artifact;
- a failed tier is never recorded as admitted;
- strict JSON rejects NaN/Infinity;
- final publication is atomic; and
- output records the highest fully completed tier, not the last attempted tier.

- [ ] **Step 2: Implement pure classification and atomic JSON**

```python
def classify_capacity_tiers(results, requested):
    admitted = []
    for count in requested:
        result = results[str(count)]
        if result["status"] != "passed":
            return {"highest_admitted": admitted[-1] if admitted else None,
                    "first_failed": count, "tiers": results}
        admitted.append(count)
    return {"highest_admitted": admitted[-1], "first_failed": None,
            "tiers": results}
```

Write to a temporary sibling with
`json.dump(payload, handle, allow_nan=False, sort_keys=True, indent=2)`, call
`handle.flush()` and `os.fsync(handle.fileno())`, then `os.replace`.

- [ ] **Step 3: Implement subprocess-isolated workers**

Each worker imports JAX only after `CUDA_VISIBLE_DEVICES` is already fixed,
constructs one exact tier, runs one full probe update, samples peak memory with
`nvidia-smi`, and publishes its tier JSON. OOM or compiler failure must exit
nonzero; the parent records stderr/stdout paths and stops.

- [ ] **Step 4: Run CLI tests and adjacent runner tests**

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest \
  tests/test_g1_rmr_probe.py tests/test_g1_tracking_runner.py -q
```

- [ ] **Step 5: Commit**

```bash
git add tools/probe_g1_rmr_capacity.py tests/test_g1_rmr_probe.py
git commit -m "feat: add G1 RMR capacity ladder"
```

---

### Task 6: Matched SHAC and PPO-Style Gradient Diagnostic

**Files:**
- Modify: `src/algorithms/shac/g1_rmr_probe.py`
- Create: `tools/compare_g1_rmr_gradients.py`
- Modify: `tests/test_g1_rmr_probe.py`

**Interfaces:**
- `collect_matched_rollout(state, keys)` returns stopped trajectory arrays plus
  differentiable SHAC inputs.
- `shac_policy_gradient(state, rollout_inputs)`
- `ppo_style_policy_gradient(state, stopped_trajectory)`
- `compute_gae(rewards, dones, terminals, values, bootstrap_values, gamma, gae_lambda)`
- CLI accepts `--num-envs`, `--horizon 12`, `--batches 8`, `--seed 42`,
  `--candidate-global-norm`, and `--output`.

- [ ] **Step 1: Write a failing closed-form PPO score-function test**

For a scalar Gaussian mean `mu`, fixed sampled action `a`, and fixed advantage
`A`, prove:

```text
d[-log_prob(a | mu, sigma) * A]/dmu = (mu - a) * A / sigma^2
```

Also assert zero gradient with respect to the trajectory action and simulator
state inputs.

- [ ] **Step 2: Implement stopped-data PPO-style loss**

```python
actions = jax.lax.stop_gradient(trajectory.actions)
advantages = jax.lax.stop_gradient(normalize(compute_gae(
    trajectory.rewards,
    trajectory.dones,
    trajectory.terminals,
    trajectory.values,
    trajectory.bootstrap_values,
    gamma=0.99,
    gae_lambda=0.95,
)))
old_log_prob = jax.lax.stop_gradient(
    gaussian_log_prob(actor_params, normalized_obs, actions)
)
new_log_prob = gaussian_log_prob(candidate_params, normalized_obs, actions)
ratio = jnp.exp(new_log_prob - old_log_prob)
surrogate = -jnp.mean(ratio * advantages)
return surrogate - 0.005 * gaussian_entropy(candidate_params.log_std)
```

At identical candidate/data-collection parameters, assert ratios equal one and
the clipping branch is inactive.

- [ ] **Step 3: Write failing matched-key and online-summary tests**

Prove each SHAC/PPO pair receives byte-identical phases, reset noise,
observation noise, Gaussian epsilon, and initial parameters; prove different
batch indices receive disjoint folded-in keys.

- [ ] **Step 4: Implement eight-batch sequential comparison**

For every batch:

- calculate SHAC and PPO-style gradients;
- update Task-3 online moments;
- accumulate pooled cross-dot products and per-layer geometry;
- apply each gradient with the same requested global parameter-step norm to the
  identical actor; and
- evaluate both candidates plus baseline on caller-fixed held-out rollout keys.

Publish pooled cosine, norm ratio, trace variance, SNR, per-layer summaries,
held-out return deltas, compile/execute time, peak memory, and all provenance.

- [ ] **Step 5: Run focused tests and commit**

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest \
  tests/test_g1_rmr_probe.py tests/test_shac_batch_gradients.py -q
git add src/algorithms/shac/g1_rmr_probe.py \
  tools/compare_g1_rmr_gradients.py tests/test_g1_rmr_probe.py
git commit -m "feat: compare SHAC and PPO gradient estimators"
```

---

### Task 7: Standalone Checkpoint Evaluation

**Files:**
- Modify: `tools/evaluate_g1_tracking.py`
- Modify: `tests/test_g1_tracking_evaluator.py`

**Interfaces:**
- Add `--random-rmr-checkpoint PATH`.
- Checkpoint contains actor parameters and actor normalizer only for
  deterministic inference; evaluator applies the actor mean.

- [ ] **Step 1: Write failing checkpoint exclusivity and inference tests**

Assert the new input is mutually exclusive with generic Flax checkpoint,
pretrained RMR checkpoint, action tape, full-pretrained-RMR mode, and residual
composition. Load a synthetic small checkpoint through a factored loader and
compare its action with `apply_rmr_mlp` on the same normalized observation.

- [ ] **Step 2: Implement the dedicated loader and mean-action path**

Do not infer checkpoint kind from tree shape. Require an explicit schema tag
`g1_random_rmr_shac_v1`, exact actor input/action dimensions, finite leaves, and
a compatible normalizer.

- [ ] **Step 3: Run evaluator tests and commit**

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest \
  tests/test_g1_tracking_evaluator.py tests/test_rmr_training_policy.py -q
git add tools/evaluate_g1_tracking.py tests/test_g1_tracking_evaluator.py
git commit -m "feat: evaluate random RMR checkpoints"
```

---

### Task 8: Verification, Registration, and Capacity Execution

**Files:**
- Create in research repo: `research/hypotheses/H-G1-027.yaml`
- Create in research repo: `research/experiments/E-20260805-098.yaml`
- Create after capacity evaluation: `research/experiments/E-20260805-099.yaml`
- Modify after execution: `research/state/current.yaml`
- Regenerate related Obsidian experiment/hypothesis projections.

**Interfaces:**
- Code commit must be clean and pushed before registration.
- Capacity artifact is the sole result source for the first experiment.

- [ ] **Step 1: Run the complete focused software gate**

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest \
  tests/test_rmr_training_policy.py \
  tests/test_g1_tracking_training_distribution.py \
  tests/test_shac_batch_gradients.py \
  tests/test_g1_rmr_probe.py \
  tests/test_g1_tracking_evaluator.py \
  tests/test_rmr_policy.py \
  tests/test_g1_tracking_environment.py \
  tests/test_g1_tracking_runner.py -q
git diff --check
```

Expected: all pass and the code worktree is clean after committing.

- [ ] **Step 2: Preregister the one-GPU capacity experiment**

The outcome map must distinguish:

- highest admitted tier with every gradient/state finite;
- first capacity failure after at least one admitted tier;
- no admitted tier;
- invalid execution or incomplete artifact.

Pin code commit, absent dirty patch, environment lock, model/reference hashes,
solver `4/5`, seed 42, one GPU, tier order, horizon 12, exact command, and a
bounded wall time. Validate and dry-run:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260805-098 --dry-run
```

- [ ] **Step 3: Execute and curate the capacity ladder**

```bash
uv run python -m tools.runexp E-20260805-098
```

Verify the final strict JSON, tier artifacts, hashes, exact environment counts,
full update completion, finite leaves, peak memory, and logs before selecting a
registered outcome.

- [ ] **Step 4: Preregister and execute the matched gradient diagnostic**

Use the smallest admitted tier justified by the capacity result. Pin eight
matched batches and the held-out key set. The outcome map must distinguish:

- useful/stable SHAC direction;
- useful/stable PPO-style direction only;
- both useful;
- both variance-limited;
- nonfinite or invalid execution.

- [ ] **Step 5: Curate registry and vault**

Update experiment result, hypothesis evidence, and current state from evaluated
artifacts; rebuild the vault; validate; commit only owned files; push both code
and research branches.

- [ ] **Step 6: Select the long-training branch**

Choose the smallest batch whose finite candidate SHAC update improves held-out
return. If no admitted SHAC tier does, follow the registered estimator/horizon
decision rather than launching an arbitrary long run. Write the next
implementation/experiment plan for full training from the evidence.
