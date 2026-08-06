# Random-Weight RMR G1 Training Through Differentiable Physics

## Goal

Train a closed-loop Unitree G1 motion-tracking controller from randomly
initialized policy weights using gradients through the MJX simulator. The task
must remain the validated RMR-style motion-tracking task. No PPO actor,
demonstration policy, action tape, behavior-cloning target, frozen source actor,
or pretrained residual may influence training or inference.

Success requires a replay-free MJX rollout and video in which the learned
standalone policy visibly tracks the reference motion without a strict terminal.
Collocation remains out of scope until this succeeds.

## Authority and the Earlier Mismatch

The authoritative RMR training inputs are:

- checkpoint run configuration
  `/home/ubuntu/projects/rmr_tracking/logs/rsl_rl/g1_flat/2026-06-10_11-11-49_walk_win137_212/params/agent.yaml`;
- environment configuration beside it at `params/env.yaml`;
- the validated MJX task and positive-control evidence from
  `E-20260805-051` and `E-20260805-060`; and
- the RMR motion and G1 model hashes already pinned by the research registry.

`E-20260805-069` and `E-20260805-070` were valid tests of the Open-DiffLoco
Go2 policy architecture on the corrected G1 task, but they were not a strict
random-weight RMR training test. They used:

- a `512/256/128` LayerNorm actor instead of RMR's
  `2048/2048/1024/1024/512/512` ELU actor;
- an exactly zero action head instead of an ordinary nonzero linear
  initialization;
- exact, unperturbed, uniformly sampled reference states;
- 64 or 256 environments with horizon 12 instead of an effective batch of
  4096 environments with horizon 24; and
- fixed action noise of `0.05` or `0.5`, rather than the RMR policy's
  trainable Gaussian exploration initialized at standard deviation `1.0`.

The new path corrects the policy, initialization, normalization, reset,
observation-noise, and exploration mismatches. Environment count and
differentiable horizon are algorithm-specific: they begin from the proven SHAC
scale and are increased only when measured capacity or gradient quality
requires it. This does not reinterpret the earlier failures.

## Selected Design

### Actor and critic

The actor is:

```text
154 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 29
```

The critic is:

```text
286 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 1
```

Every hidden layer uses ELU. There is no LayerNorm and no actor output
activation. Each linear weight and bias is sampled from
`Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`, matching the distribution used by
PyTorch `nn.Linear.reset_parameters`. JAX and PyTorch PRNG streams are not
expected to produce byte-identical samples; the initialization distribution,
shapes, dtypes, and seed are evidence.

The actor owns a trainable 29-dimensional log standard deviation initialized to
zero, so its initial Gaussian action standard deviation is exactly one. Training
uses reparameterized actions:

```text
action = actor_mean(normalized_observation) + exp(log_std) * epsilon
```

The entropy coefficient is `0.005`, matching RMR. Evaluation uses the
deterministic actor mean. The log representation is the sole numerical
parameterization difference from RSL-RL's direct standard-deviation parameter;
it preserves the same initial action distribution while preventing an invalid
nonpositive standard deviation.

Both actor and critic observation normalizers begin from zero mean and unit
variance, update only from collected training observations, and use RMR's
normalization epsilon `1e-2`. No statistics are loaded from the PPO checkpoint.
Compilation must not advance a normalizer, environment, optimizer, sampler, or
PRNG state.

### Task and reset distribution

The environment retains the validated source-order RMR observation, action,
reward, termination, 50 Hz control, reference timing, solver `4/5`, nominal
actuator gains, action scales, and unbounded action support.

Training resets sample reference phases with RMR's failure-adaptive sampler:

- one-frame adaptive kernel;
- uniform ratio `0.2`;
- exponential failure update coefficient `0.001`; and
- an initially uniform distribution.

Each reset applies the recorded RMR perturbations:

- root position: `x/y ±0.02 m`, `z ±0.005 m`;
- root roll/pitch/yaw: `±0.1 rad`;
- root linear velocity: `x/y ±0.25 m/s`, `z ±0.1 m/s`;
- root angular velocity: roll/pitch `±0.26 rad/s`, yaw `±0.39 rad/s`; and
- joint position: `±0.05 rad`, clipped to the same soft limits.

Joint velocities start from the selected reference frame. Actor observations
receive the recorded independent uniform corruption:

- anchor orientation `±0.05`;
- base angular velocity `±0.2`;
- joint position `±0.01`; and
- joint velocity `±0.5`.

Critic observations remain uncorrupted.

The first learning gate uses the nominal MJX plant. After nominal visible
tracking succeeds, a separately registered robustness stage introduces the
source RMR startup dynamics randomization and interval pushes. This ordering
does not weaken the final training target: it separates “can simulator gradients
learn the motion?” from “can the learned policy tolerate the source
randomization?” and prevents domain randomization from hiding the first failure
mechanism.

### Differentiable learning

PPO is replaced by SHAC:

- the first actor objective is the 12-step differentiated return used by
  successful Open-DiffLoco SHAC, plus the RMR entropy term;
- `gamma=0.99`, `lambda=0.95`, actor learning rate `1e-3`, and aggregate global
  gradient clipping at norm `1.0`;
- the critic uses the same architecture and the existing SHAC TD(lambda)
  target/target-network method;
- the actor and critic start from random parameters; and
- no PPO likelihood ratio, clipping objective, KL schedule, or checkpoint is
  used.

This is the intended meaning of “same RL task, otherwise with DiffSim”: task,
policy family, initialization distribution, exploration, and data distribution
match RMR. Batch size, differentiable horizon, estimator, and optimizer data
flow follow SHAC because PPO's 4096-environment sampling regime is not evidence
for the variance or memory needs of simulator gradients.

## One-GPU Capacity and Batch Selection

The existing trainer materializes one complete policy gradient per environment.
For the 8,463,901-parameter actor, 4096 float32 gradient copies alone would
require about 139 GB before simulator intermediates. That implementation cannot
be used to decide whether a larger batch is scientifically useful.

The dedicated RMR trainer instead differentiates a batch-mean loss directly and
produces one aggregate actor gradient. The first registered execution is a
single-L40S capacity ladder with the exact random actor, task, resets,
observation noise, exploration, and horizon 12. It tries ordered environment
counts `256`, `512`, `1024`, and `2048`, stopping after the first capacity
failure or when the registered wall-time limit is reached. Each admitted tier
must complete one actor update and its complete critic-update boundary; a reset
or forward-only result is not a passed tier.

The capacity ladder answers only what fits. It does not select the largest
possible batch by default. Afterward, the smallest admitted batch whose
independent gradient estimates are useful on fixed held-out rollouts becomes
the training batch. Larger batches are justified only by measured disagreement
or held-out update failure. Gradient accumulation or multi-GPU all-reduce is a
later response to that evidence, not prerequisite infrastructure.

### Matched SHAC/PPO gradient diagnostic

At the selected one-GPU capacity, a separate registered diagnostic compares two
policy-gradient estimators without changing simulator, task, policy, data, or
randomness:

1. **SHAC pathwise gradient:** reparameterize the Gaussian action and
   differentiate return through MJX.
2. **PPO-style score-function gradient:** use the same sampled MJX trajectory,
   stop gradients through actions and simulator state, form RMR's normalized
   GAE advantages, and differentiate the on-policy log-probability surrogate.
   At the data-collection policy the likelihood ratio is one, so PPO clipping
   is inactive; the result is the actual first on-policy PPO policy-gradient
   direction plus the same entropy term.

This is intentionally not a comparison against a separately executed PhysX
rollout, which would confound gradient estimator and simulator. The pretrained
RMR actor is never loaded. Both estimators start from the identical random actor
and critic and consume caller-owned matched phase samples, reset
perturbations, observation corruption, and Gaussian action epsilon.

Eight disjoint matched batches are evaluated sequentially without retaining
eight full gradient trees. Online summaries record:

- aggregate and per-layer gradient norms;
- pairwise cosine to each estimator's pooled mean;
- trace variance and signal-to-noise ratio
  `||mean(g)||^2 / mean(||g_i - mean(g)||^2)`;
- cosine and norm ratio between the pooled SHAC and PPO-style gradients;
- compile time, execution time, and peak device memory; and
- held-out return changes after equal-global-norm SHAC and PPO-style candidate
  steps from the identical parameters.

Neither estimator is declared ground truth from cosine alone. The practical
gate is whether its equal-size candidate update improves fixed held-out
rollouts. Gradient variance and cross-estimator direction explain that result
and determine whether increasing the environment count is justified.

The preflight must establish:

1. On a finite small batch, direct mean-loss differentiation agrees with the
   mean of legacy per-environment gradients before clipping.
2. Every capacity tier consumes disjoint registered PRNG keys and RSI phases.
3. The actor and critic gradients, updated parameters, and all
   environment states are finite.
4. The exact highest completed environment count and peak device memory are
   recorded without changing the registered task.
5. Two independent actor gradients at every admitted tier have their cosine,
   norms, and update ratios recorded, and the corresponding clipped updates are
   evaluated on caller-fixed held-out rollouts.
6. At the selected tier, eight matched SHAC/PPO-style estimates produce finite
   online variance, direction, cost, and held-out-update summaries.

No long training run starts until this preflight passes.

## Implementation Boundaries

The implementation is a dedicated G1 RMR-from-scratch path rather than another
set of branches inside the existing Go2-oriented SHAC function:

- a small pure-JAX RMR MLP module owns initialization and application;
- G1 training-reset helpers own perturbations and adaptive phase sampling;
- a dedicated SHAC trainer owns aggregate mean gradients and the one-GPU
  capacity boundary; and
- the CLI binds the exact configuration and artifact directory.

Existing Go2 training, generic G1 SHAC, pretrained full-policy fine-tuning, and
residual paths remain unchanged. Shared checkpoint and evaluation helpers may be
reused only when their contracts already fit.

## Test and Evidence Gates

Implementation tests must prove:

- exact actor/critic layer shapes, parameter count, nonzero output head, and
  initialization bounds;
- deterministic actor application and Gaussian sampling from caller-owned
  epsilon;
- online normalizer initialization and updates without compile-time mutation;
- reset perturbations remain within every registered range;
- adaptive sampling starts uniform and increases failed-bin probability;
- actor corruption touches only the registered fields and the critic remains
  clean;
- direct batch gradients agree with legacy averaged gradients on a finite toy
  boundary;
- aggregate batch gradients equal an explicit mean-gradient reference on a
  small problem; and
- the PPO-style diagnostic stops simulator/action gradients and reproduces a
  closed-form score-function gradient on a toy Gaussian policy;
- matched-estimator online moments equal explicit stored-gradient moments on a
  small tree;
- checkpoints reload into the standalone deterministic evaluator.

Every GPU run receives a schema-v1 experiment registration with pinned code,
environment, model, reference, command, device count, seed, wall-time budget,
outcome map, and required artifacts.

Training checkpoints are selected by replay-free evaluation, never by noisy
training reward alone. The strict evaluation uses phases `0/30/60/90`, the
validated `4/5` solver, deterministic actor means, and at least 60 control
steps. A visible-success claim requires:

- no terminal in the registered strict rollouts;
- finite trajectory and metric artifacts;
- material improvement from the identical random initialization;
- comparison with the pretrained RMR positive control as an unattained
  reference, not as a training input; and
- a video and first/middle/final montage showing the actual simulated G1 beside
  the reference.

## Decision Branches

- **A capacity tier fails for memory only:** retain the highest fully completed
  tier and do not build distributed execution unless its gradients fail the
  held-out usefulness gate.
- **Gradients are nonfinite:** localize the first environment/time/component;
  do not start training or replace invalid values without a registered causal
  repair.
- **Finite exact-contract training improves and walks:** reproduce with another
  seed, then add RMR dynamics randomization and evaluate sim-to-sim.
- **Finite exact-contract training remains outside the walking basin:** retain
  the exact actor/task and preregister a horizon curriculum
  `1 -> 4 -> 12 -> 24`, or increase the effective batch only if recorded
  gradient disagreement supports it; do not fall back to PPO initialization.
- **PPO-style directions improve held-out returns but SHAC directions do not:**
  localize the pathwise gradient disagreement by horizon and contact phase
  before changing the task or policy.
- **Both estimators have poor signal-to-noise at the admitted batch:** increase
  effective batch through accumulation before beginning a long run.
- **An early checkpoint walks and later checkpoints regress:** select only by
  the fixed strict evaluator and register the duration/early-stopping question.

At no point does residual learning or PPO initialization count as
from-scratch success.
