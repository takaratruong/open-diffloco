# G1 Adaptive-Sampling And CAGrad Continuations Design

## Objective

Test two distinct explanations for E-20260810-006's phase redistribution without
paying for two new actors from random initialization:

1. uniform reset sampling spends too little training mass near phases where the
   carried policy fails; and
2. actor gradients from different reference regions destructively interfere
   when reduced to one scalar mean.

Run two independent continuations in parallel. Both restore the exact complete
E006 training state and retain the same model, reference, fixed 4x5 solver,
actor, critic, optimizers, normalizers, PRNG state, environments, domain
randomization, action noise schedule, effective batch, and horizon. One changes
only the reset-phase distribution. The other changes only actor-gradient
aggregation. Do not combine the treatments in this comparison.

E006 is the immutable parent because its exact final actor is finite but
survives only `15/49/54/17/17` frames from phases `0/100/200/300/400`.
Effective-batch scaling improved phases 0 through 200 while collapsing phase
300, so more averaging alone did not raise the robust floor.

## Alternatives Considered

### Recommended: paired exact-state continuations

Resume both treatments from the complete E006 TrainState. This directly tests
whether either mechanism repairs the observed actor and costs half as many
updates as a fresh 128-update gate. Exact continuation also preserves the
optimizer and critic state that shaped the failure.

### Fresh matched training from seed zero

Fresh actors provide the cleanest from-scratch comparison, but repeat E006's
already-paid 128 updates before either treatment reaches the current failure
boundary. This is unnecessary for the first discriminator. A passing treatment
must later reproduce from a fresh seed before supporting a general claim.

### Combined adaptive sampling plus CAGrad

The combination could be stronger, but a positive result would not identify
whether improved state coverage, conflict-aware gradients, or their
interaction caused it. A combined successor is authorized only if at least one
isolated arm advances.

## Shared Scientific Contract

Restore the exact E006 final checkpoint at step `786,432`, SHA-256
`1c1d657333e5ede6e402b494deb9d6e9d94771efa258497dc8d16a79b8905cef`.
Continue for exactly 64 actor updates. Preserve physical `num_envs=256`, two
sequential gradient shards, effective batch 512, horizon 12, and therefore add
exactly `64 * 512 * 12 = 393,216` transitions, ending at step `1,179,648`.

The resumed action-noise schedule remains the parent's schedule; it is not
restarted. The actor and critic learning rates, optimizer moments, target
critic, observation normalizers, environment states, domain randomization,
reference-reset perturbation scale, rewards, termination rules, controller,
and plant remain unchanged. Each arm gets its own immutable runner and output
root. Scientific settings are not exposed as command-line overrides.

Archive fixed checkpoints after 32 and 64 continuation updates. Evaluate both
checkpoints under the same deterministic phase-grid protocol. Select within an
arm lexicographically by maximum minimum survival, then median survival, then
mean survival, then earliest checkpoint. Do not inspect or select any other
checkpoint.

## Arm A: Adaptive Failure-Biased Reference Resets

### Sampling rule

Use the existing training-only phase-sampler boundary, but make its uniform
component a literal mixture rather than an additive pseudocount. Partition the
reference timeline into the established one-second bins. Maintain an EMA of
true-terminal counts by terminal reference phase with `alpha=0.001`.

At each reset, sample a bin from:

```text
0.5 * Uniform(all bins) + 0.5 * Normalize(failure EMA)
```

When the failure EMA is all zero, use the uniform distribution. Sample a frame
uniformly inside the selected bin. This retains a 50-percent uniform floor even
after many failures and prevents the treatment from training only the current
weakest region.

The treatment changes only the selected reference phase. After phase
selection, apply the parent's unchanged noisy reference-state initialization
and domain randomization. Do not initialize from the already-collapsed terminal
state and do not use a carried-state bank. This distinguishes phase-local
difficulty from recovery out of an arbitrary, potentially unrecoverable state.

Update the failure EMA once per complete effective-512 actor update using true
terminal phases from both physical shards. Resets inside the next actor update
consume the previously completed EMA, avoiding within-update order dependence.
The EMA and current probabilities must be checkpointed with the treatment and
restored exactly if execution is resumed.

### Telemetry and validity

At every existing diagnostic interval record bin counts, failure EMA, sampling
probabilities, terminal counts, the realized uniform/failure-biased reset
fractions, and the usual finite-gradient statistics. Probabilities must be
finite, nonnegative, sum to one within float tolerance, and retain at least
half of the literal uniform mass. Missing telemetry, a distribution mismatch,
or a change to reset perturbations or randomization makes the run invalid.

## Arm B: Five-Phase CAGrad Actor Aggregation

### Task gradients

Keep reset phases uniformly sampled. Map every environment's actor-rollout
start phase into five equal reference-timeline bins. Within each physical shard,
compute finite-aware gradient sums and contributor counts for every bin. Merge
the two shards before forming one mean actor gradient per bin, so CAGrad sees
the complete effective batch rather than two independently combined updates.

The critic remains the parent's ordinary uniform effective-512 mean. CAGrad
changes only the actor-gradient reduction.

### CAGrad rule

Let `g_i` be the five mean actor gradients, `g_0` their uniform mean, `G` their
5-by-5 Gram matrix, and `b` the uniform five-vector. Use standard CAGrad with
conflict-aversion coefficient `alpha=0.5`:

```text
c = alpha * ||g_0||
w* = argmin_{w in simplex} w^T G b + c * sqrt(w^T G w + eps)
g_w = sum_i w*_i g_i
g = (g_0 + c * g_w / (||g_w|| + eps)) / (1 + alpha^2)
```

Solve the five-variable convex simplex problem in JAX with a fixed 32-step
Frank-Wolfe loop. Runtime code must not depend on SciPy. A test-only SciPy SLSQP
oracle verifies the fixed solver's objective and combined direction on seeded
positive-semidefinite Gram matrices before GPU execution.

All five bins must be occupied for the registered treatment. Any missing task,
nonfinite Gram entry, nonfinite weight, simplex violation, or nonfinite combined
update makes the run invalid rather than silently reverting to scalar means.

### Telemetry

Record per-bin counts, losses, gradient norms, the full 5-by-5 cosine matrix,
CAGrad weights, optimizer objective, Frank-Wolfe dual gap, combined-gradient
norm, and uniform-mean/CAGrad cosine. This establishes whether phase conflict
was actually present and whether the treatment altered the update direction.

## Code Boundaries

Keep both features default-off.

- Extend the focused training-distribution module and G1 reset seam for the
  adaptive sampler. Do not change exact evaluation reset behavior.
- Add a focused `src/algorithms/shac/cagrad.py` containing pure bin reduction,
  Gram construction, simplex optimization, and gradient combination functions.
- Integrate the two options into SHAC only at reset selection and actor-gradient
  aggregation respectively. Avoid a second trainer or duplicated actor loss.
- Add two immutable continuation runners that copy E006's effective-512 resume
  contract and enable exactly one treatment each.

The two pure modules and their tests can be developed in parallel. Integration
through the shared SHAC algorithm and environment seams is reviewed centrally,
then both registered experiments launch concurrently on distinct free GPUs.

## Testing And Verification

Use test-driven development at these boundaries:

1. A zero failure EMA is exactly uniform; a concentrated failure EMA creates
   the literal 50/50 mixture and never removes the uniform floor.
2. Adaptive reset phase selection preserves the parent's perturbation,
   randomization, observation-history, and continuation-RNG contracts.
3. Terminal phases from both shards update one order-independent EMA, which is
   present in saved and restored treatment checkpoints.
4. Identical phase gradients make CAGrad collinear with the uniform mean;
   conflicting seeded gradients match the SciPy oracle within registered
   tolerance.
5. Five-bin aggregation across two shards equals aggregation over their
   concatenated population.
6. Default-off execution reproduces the existing effective-512 aggregation and
   reset behavior.
7. Each dedicated runner differs from the E006 continuation contract only by
   its named treatment and the mechanically required final step/checkpoints.

Run focused unit and contract tests, Ruff, Python compilation, the canonical
SHAC smoke, exact resume checks, research-registry validation, and guarded dry
runs before launching either GPU experiment.

## Evaluation And Decision Gate

Evaluate exact starts `0`, `100`, `200`, `300`, and `400` with nominal physics,
no replay, no teleportation, and no post-terminal reset evidence.

An arm advances only when:

- training and both fixed evaluations are complete and finite;
- its selected checkpoint has minimum survival at least 19 and median survival
  at least 48; and
- no phase loses more than five frames relative to E006's immutable
  `15/49/54/17/17` vector.

If one arm advances, retain it and stop the other. If both advance, compare
their selected vectors and register a separate combined-treatment successor.
If neither advances, stop both mechanisms at this budget and return to a
representation or longer-credit-assignment change. A pass is only an early
continuation gate; it does not establish fresh-seed reproducibility, full
suffix completion, transfer, or hardware readiness.
