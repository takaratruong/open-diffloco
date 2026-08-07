# SHAC Gradient Accumulation Design

## Objective

Test whether larger effective pathwise-gradient batches close
E-20260805-098's remaining five-step G1 tracking failure without materializing
512 or 1,024 per-environment gradients on one L40S. The implementation must
preserve the existing 256-environment, horizon-12 SHAC estimator and perform
one optimizer update per effective batch.

E-20260806-000 and E-20260806-001 establish the resource boundary: direct
`num_envs` values 512 and 1,024 fail before update one on 42.74 GiB and
85.47 GiB allocations. They do not answer whether larger effective batches
improve learning.

## Considered Approaches

### In-JIT sequential microbatch scan — selected

Keep `num_envs` as the physical microbatch size and add a positive integer
`gradient_accumulation_steps`. Store the complete effective population's
environment states, random inputs, and resulting trajectories, but execute
per-environment differentiation on one microbatch at a time inside a JAX
scan. Average gradients before optimizer updates.

This keeps device execution asynchronous, avoids host/device synchronization,
and compiles one fixed 256-environment differentiation body. Trajectories and
environment state are far smaller than the materialized per-environment
parameter gradients that caused the OOM.

### Host-side accumulation

Expose separate compiled gradient and update functions and loop over shards in
Python. This is easy to inspect but would require repeated device round trips,
split actor/critic state transitions, and up to 64 critic-gradient calls per
effective update. It creates more coupling and is not selected.

### Multi-device `pmap`

Run one 256-environment shard per GPU and all-reduce gradients. This could make
large effective batches faster, but it adds distributed compilation, state
sharding, collective failure, and checkpoint portability boundaries before
the scientific value of larger batches is known. It is deferred until
sequential accumulation shows a useful learning effect.

## Interface and Semantics

Add `gradient_accumulation_steps: int = 1` to `train()` and
`--gradient-accumulation-steps` to the native RMR runner. Reject booleans,
non-integers, and values below one. Record both the physical `num_envs` and:

```text
effective_num_envs = num_envs * gradient_accumulation_steps
steps_per_actor_update = effective_num_envs * unroll_length
```

`total_steps`, curriculum progress, learning-rate schedules, checkpoint
intervals, logging, and `TrainState.step` use effective environment steps.
Factor one must retain the current behavior and checkpoint semantics.

The initial environment population contains `effective_num_envs` independent
resets produced by one ordered `jax.random.split`. Within `train_step`, every
batched environment-state leaf and every stochastic input is reshaped to
`(gradient_accumulation_steps, num_envs, ...)`. Random action noise, pushes,
terrain innovations, and difficulty masks are generated once over the full
effective population before reshaping; accumulation therefore does not reuse
random samples across shards.

## Actor Accumulation

For every microbatch, hold actor parameters, target-critic parameters, and
both normalizers fixed. Run the existing horizon rollout and materialize its
per-environment actor gradients for only `num_envs` environments.

Apply the existing finite-rollout rejection and per-environment global-norm
clip before accumulating. For equal-size shards, the mean of shard means is
the exact full effective-batch mean. Sanitize the accumulated tree, compute
one aggregate norm, and perform exactly one actor optimizer update.

Retain all shard trajectories and final environment states for critic
training, normalizer updates, metrics, and the next effective update. The
implementation must not perform an actor update inside the microbatch scan.

Gradient diagnostics aggregate across the effective population:

- finite fraction is the weighted mean of shard finite fractions;
- maximum raw norm is the maximum of shard maxima;
- raw median is computed from the complete concatenated per-environment norm
  vector, not by averaging shard medians.

The gradient aggregation helper will therefore expose per-environment raw
norms and finite flags in addition to the current summary.

## Critic Accumulation

Each of the existing 16 critic iterations remains one optimizer update. At a
given iteration, hold the current critic parameters fixed, compute
per-environment critic gradients shard by shard over the stored trajectories,
apply the same finite filtering and per-environment norm clip, average across
all shards, and then perform one critic optimizer update. Do not update the
critic between microbatches.

The target critic is updated once after all 16 accumulated critic iterations,
matching the current control flow.

## Normalizers, State, and Metrics

Update actor and critic normalizers once from every observation in the
effective population. Reassemble final environment states in their original
flat ordering and preserve them across actor updates. Metrics are means over
all microbatches and environments.

Warm-up may update only normalizers, as in the existing trainer. Factor one
must keep the current committed-array cache signature and must not introduce a
second real-step compilation.

Checkpoint and hparams provenance must include
`gradient_accumulation_steps`, `effective_num_envs`, and
`steps_per_actor_update`. Resumption must reject a changed accumulation factor
or effective population shape rather than silently resetting environment
state semantics.

## Failure Handling

- Reject invalid accumulation values before creating an environment or output
  directory.
- A non-finite environment gradient is removed as one whole environment before
  cross-shard averaging, exactly as in the existing estimator.
- A shard cannot be retried, skipped, resized, or reweighted.
- Any OOM, timeout, missing shard, mismatched effective step, non-finite
  checkpoint, or absent diagnostic is an invalid execution.
- Do not fall back to multiple Adam updates or direct larger `num_envs`.

## Verification Gates

### Unit and factor-one compatibility

1. Accumulated equal-size shard means match an explicit concatenated mean after
   per-environment clipping, including non-finite and extreme gradients.
2. Full-population finite fraction, median, and maximum match explicit
   concatenation.
3. Runner validation and transport are exact.
4. Factor one produces the same update count, step count, parameters,
   optimizer state, normalizers, environment state, and metrics as the
   unaccumulated path within dtype-appropriate tolerance.

### Small actual-MJX batch-shape sensitivity diagnostic

Compare one monolithic four-environment update with two accumulated
two-environment shards using the same seed and pre-sampled ordered randomness,
but do not treat it as an estimator-parity authority. The comparison changes
the physical `vmap` width from four to two. Contact-rich MJX trajectories and
their pathwise gradients can diverge under that numerical batching change even
when the shard aggregation algebra is exact.

Record step, shape, finiteness, parameter, optimizer, normalizer, and
environment-state differences. A failure diagnoses physical-batch numerical
sensitivity; it must not be converted into a pass by relaxing thresholds.
Production admission instead requires all of the following:

- factor one retains the committed physical-256 path;
- pure clipped-gradient tests prove that equal-size shard means equal the
  concatenated effective-population mean;
- every accumulated shard uses physical width 256, exactly matching E098;
- the production resource gate below publishes a complete finite update.

### Production resource gate

Run one effective-512 update as two sequential 256-environment shards on one
L40S. It must compile, remain below device memory, execute one actor plus all
16 critic updates, publish complete finite diagnostics, and advance exactly
6,144 environment steps. Failure stops the accumulation branch before long
training.

An effective-1,024 technical gate may run concurrently as four sequential
physical-256 shards. It must satisfy the analogous 12,288-step contract before
the factor-four learning arm is admitted.

## Learning Experiments

E-20260805-098 is the immutable factor-one control: physical/effective batch
256, horizon 12, 128 actor updates, and 393,216 transitions.

After both verification gates pass, run concurrently on separate GPUs:

| Accumulation | Physical batch | Effective batch | Actor updates | Transitions |
|---:|---:|---:|---:|---:|
| 2 | 256 | 512 | 128 | 786,432 |
| 4 | 256 | 1,024 | 128 | 1,572,864 |

Preserve E098's seed, compact actor, random head, absence of LayerNorm, horizon,
learning rate, noise, bootstrap scale, task, rewards, termination, critic,
solver, model, reference, and evaluation protocol.

Evaluate actor updates 10, 20, 30, 40, 50, and 128 at phase zero. Select by
strict survival, terminal absence, reward, then registered tracking errors;
evaluate the selected checkpoint at phases 30, 60, and 90 with trajectory,
video, and first/middle/final montage.

Run accumulation factor 8 only if factor 4 exceeds E098's 55-step phase-zero
survival or reaches 60/60. If neither factor 2 nor factor 4 exceeds 55, reject
larger effective batch as the next mechanism and pivot to temporal stability
or deterministic checkpoint selection.
