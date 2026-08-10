# G1 Phase-Robust Actor Loss Design

## Objective

Test whether the canonical G1 SHAC actor fails because equal averaging lets
conflicting reference-phase gradients cancel or neglect the weakest phases.
The experiment changes only the relative magnitude of per-environment actor
gradient contributions. It does not change phase sampling, reset states,
critic training, rewards, horizon, batch size, optimizer, or evaluation.

The parent comparison is E-20260810-004. Its fixed-horizon-24 actor completed
393,216 finite transitions, but replay-free survival changed from the
horizon-12 vector `14/43/47/61/19` to `15/42/47/19/41`. The collapse at phase
300 alongside the gain at phase 400 is consistent with phase-gradient conflict,
but does not prove it. This bounded experiment retires that uncertainty.

## Alternatives Considered

### Recommended: detached bounded phase-bin weighting

Partition the 499 valid starting phases into five equal timeline bins. Compute
the mean actor loss in each occupied bin, derive bounded weights from detached
bin losses, and use those weights only when aggregating per-environment actor
gradients. This changes gradient direction while preserving the mean update
scale and avoids differentiating through the weighting rule.

### Hard CVaR over the worst bins

Aggregate only the worst one or two phase bins. This targets the failure more
aggressively but discards most of a 256-environment batch and makes bin ranking
noise capable of flipping the complete update direction. It is not selected
for the first bounded test.

### Fixed weights from E004 evaluation

Hard-code larger weights for phases 0 and 300. This is simple but overfits five
evaluation starts, leaks the parent result into training, and cannot adapt when
the weakest phase changes. It is rejected.

## Scientific Contract

Start from the fresh seed-0 initialization and complete exactly 393,216
transitions with the E004 contract: 256 environments, 24 differentiable steps,
fixed-scan `g1-4x5`, the canonical actor and critic, actor learning rate
`5e-3`, critic learning rate `5e-4`, terminal critic bootstrap, residual-action
semantics, observation history, noise, domain randomization, curriculum, and
checkpoint interval.

Keep reset phases uniformly sampled. Do not invoke the existing adaptive phase
sampler. Keep the critic loss and critic-gradient aggregation uniform. Do not
resume E004.

## Weighting Rule

Capture each environment's phase at the start of its 24-step actor rollout.
Map phase `p` to:

```text
bin = min(4, floor(5 * p / 499))
```

For every occupied bin, compute its mean per-environment actor loss. Standardize
the occupied bin means with their detached mean and population standard
deviation. When the standard deviation is below `1e-6`, use zero logits so all
weights remain exactly one. Otherwise apply softmax to the standardized losses;
higher actor loss means a harder bin.

Mix the resulting robust distribution with a uniform distribution over the
occupied bins. With robust fraction `r`, convert the distribution to relative
bin weights:

```text
bin_weight = (1 - r) + r * occupied_bin_count * softmax(standardized_bin_loss)
```

Set `r = 0.5` for this experiment. The rule then naturally bounds occupied-bin
weights to `(0.5, 3.0]` for five bins. Gather one weight per environment and
renormalize the environment weights to mean exactly one. Stop gradients through
losses, statistics, and weights.

Aggregate actor gradients as a finite-aware weighted mean. For each gradient
leaf, nonfinite elements contribute neither numerator nor denominator. If a
leaf position has no finite contributor, emit zero there, matching the current
trainer's nonfinite aggregate behavior. All-finite equal weights must reproduce
the existing unweighted mean within floating-point tolerance.

This rule does not multiply the complete loss by a constant. A uniform scalar
would mostly be normalized away by Adam and would not change the competition
between phase gradients.

## Code Boundary

Add a focused phase-weighting module under `src/algorithms/shac/` containing
pure functions for bin assignment, detached bin weights, and finite-aware
weighted gradient aggregation. The established unweighted aggregation remains
the default and byte-compatible.

Expose three opt-in trainer settings:

- `actor_phase_robust_weighting: bool = False`;
- `actor_phase_bin_count: int = 5`;
- `actor_phase_robust_fraction: float = 0.5`.

The dedicated successor runner copies the complete E004 keyword contract and
changes only `actor_phase_robust_weighting` from false to true. It exposes no
scientific override and no resume option.

## Telemetry And Error Handling

Record, every existing diagnostic interval:

- per-bin environment count;
- unweighted per-bin actor loss;
- detached per-bin relative weight;
- weighted and unweighted aggregate actor loss;
- weighted actor-gradient norm and existing finite-gradient statistics.

The hparams artifact records all three phase-robust settings. Missing or
nonfinite bin losses, weights outside `[0.5, 3.0]` before final mean
normalization, a nonfinite weighted update, a contract mismatch, or missing
telemetry makes the experiment invalid. An empty bin is excluded from the
softmax and does not itself invalidate an update.

## Testing

Use test-driven development at three risk boundaries:

1. Equal bin losses produce exact unit weights and reproduce the unweighted
   aggregate.
2. A uniquely worse bin receives the largest bounded weight while environment
   weights retain mean one and remain detached.
3. Nonfinite gradient elements are excluded from their weighted numerator and
   denominator without contaminating finite leaves.

Add a contract test proving the successor differs from E004 only by enabling
phase-robust actor weighting. Run focused SHAC aggregation and runner tests,
Ruff, Python compilation, exact diff checks, registry validation, and a guarded
dry run before GPU execution.

## Evaluation And Stop Gate

After finite completion, evaluate the final actor under nominal physics from
exact phases `0`, `100`, `200`, `300`, and `400`, without replay, teleportation,
or post-terminal reset evidence.

Advance only if all required artifacts and telemetry are valid, minimum
survival is at least `19`, and median survival is at least `48`. Otherwise stop
the phase-robust-loss branch. Passing this early gate authorizes a separately
registered continuation; it does not establish full tracking, repeatability,
sim-to-real transfer, or hardware readiness.
