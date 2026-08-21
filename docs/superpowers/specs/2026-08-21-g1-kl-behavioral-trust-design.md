# G1 KL-Calibrated Behavioral Trust Discriminator

## Purpose

Determine whether PPO-like policy proximity can turn one ordinary E023 SHAC
proposal into a componentwise-safe walking improvement. This is a bounded
discriminator, not a training sweep. A full adaptive-KL continuation is
authorized only if this discriminator finds a safe candidate.

## Scientific Question

Does backtracking one unchanged H24 SHAC update by its Gaussian action KL
produce a candidate that preserves every retained E023 replay-free phase while
improving at least one phase or the paired tracking metrics?

The uncertainty retired is whether E023's destructive closed-loop tradeoff is
primarily caused by an excessively large policy step. The result changes the
next branch as follows:

- A componentwise-safe improvement authorizes one adaptive-KL continuation at
  the measured KL target.
- KL-feasible candidates that remain behaviorally unsafe reject policy-step
  size as the next axis and return the program to state distribution or
  objective design.
- No nonzero KL-feasible candidate means the proposed trust target is too
  strict for this update and does not authorize target tuning.

## Treatment

Start from the immutable E023 update-128 checkpoint and execute exactly one
ordinary effective-512, H24, five-bin CAGrad update. Preserve E023's model,
reference, solver, exponential reward kernel, observations, RMR action-noise
schedule, reset distribution, optimizer, zero bootstrap, per-environment
gradient clipping, and seed.

Construct parameter-space candidates along the exact source-to-proposal path at
the preregistered alphas:

`1, 1/2, 1/4, 1/8, 1/16, 1/32`.

This is deterministic backtracking, not a hyperparameter sweep. No alpha may be
added after viewing results. Every candidate retains the source checkpoint's
actor normalizer bit-exactly. The one-update normalizer is not interpolated or
selected: as in PPO's frozen rollout batch, proximity is measured and deployed
under the observation transform that generated the batch.

Candidate checkpoints are compact evaluation-only records containing the
interpolated actor parameters and the bit-exact source normalizer. They omit
environment, critic, optimizer, and RNG state and therefore cannot be resumed
for training. This avoids six 428 MB copies of irrelevant and misleading
optimizer state. A successful discriminator authorizes a new optimizer
implementation; it does not promote an interpolation artifact into a training
parent.

## PPO-Style Proximity Evidence

Use the immutable 512-state E023 objective-audit bank. For each candidate,
apply the source and candidate actors to the same normalized 10-frame actor
histories using the immutable source normalizer. Treat the exploration policy
as a diagonal Gaussian with the exact E023 action-noise standard-deviation
vector.

For every state and action dimension compute

`KL(old || candidate) = (mu_candidate - mu_old)^2 / (2 * sigma^2)`.

Publish the mean per-dimension KL over all states, the per-state 95th percentile
and maximum, and the empirical log-ratio on the bank's first common epsilon
sample. A candidate is KL-feasible only when:

- mean per-dimension KL is at most `0.01`;
- per-state 95th-percentile KL is at most `0.02`;
- all means, standard deviations, KL values, and log-ratios are finite; and
- the source candidate has bit-exact zero KL.

The KL calculation is diagnostic and selection evidence. It does not replace
SHAC's pathwise physics gradient with PPO's likelihood-ratio gradient.

## Behavioral Acceptance

Evaluate the source, the full proposal, and every backtracked candidate under
the exact replay-free E023 phase grid at phases `0/25/50/75/100`. Use the same
compiled environment step, post-policy action boundary, seed, reference, and
solver as E023.

The protected source floor is `116/99/67/49/24`. A candidate is behaviorally
safe only when every survival component is at least that floor. Among candidates
that are both KL-feasible and behaviorally safe, select the largest alpha. The
selected candidate advances only if it strictly improves one survival component
or improves paired mean tracking reward and body-position error without any
registered tracking metric worsening by more than five percent.

No training metric, averaged survival score, or improvement in one phase may
compensate for regression in another phase.

## Outcomes

- `kl-trust-advances`: at least one KL-feasible candidate clears the protected
  phase floor and the advancement gate. Use its measured KL as the fixed target
  for one separately registered adaptive-KL continuation.
- `kl-trust-preserves-only`: at least one KL-feasible nonzero candidate clears
  the protected phase floor but none advances. PPO-style proximity controls
  drift but does not yet improve E023; do not run a continuation.
- `kl-feasible-behaviorally-unsafe`: at least one nonzero candidate clears the
  KL gate, but every such candidate regresses a protected phase. Stop update-
  magnitude and KL-target tuning.
- `no-nonzero-kl-candidate`: no nonzero candidate clears the preregistered KL
  gate. Do not tune the target post hoc.
- `invalid-execution`: any provenance, state identity, actor application,
  Gaussian, interpolation, one-update, phase-grid, finiteness, or artifact gate
  fails.

## Artifacts and Failure Handling

The run must publish a hash-bound preflight, the one-update checkpoint and
training validation, the candidate manifest, a KL/log-ratio table, all seven
phase grids including the source, a deterministic selection record, and the
selected replay-free video/contact sheet when a nonzero candidate is selected.

Every validator fails closed on missing keys, nonfinite values, asset drift,
unexpected checkpoints, mismatched actor structures, changed normalizers,
incorrect source floors, or disagreement between raw evidence and summaries.
Artifacts are written atomically. A failed gate produces no scientific
selection.

## Testing Boundary

Unit tests cover exact Gaussian KL/log-ratio calculations, zero-KL identity,
fixed-standard-deviation validation, interpolation identity and finiteness,
largest-safe-alpha selection, componentwise behavioral rejection, outcome
precedence, provenance, and atomic manifest validation. One CPU smoke applies
the real E023 actor to a small slice of the frozen bank before GPU execution.

The GPU run is limited to one SHAC update plus deterministic evaluation of the
six nonzero candidates and source. There is no adaptive target tuning and no
full continuation in this experiment.
