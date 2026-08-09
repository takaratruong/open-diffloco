# G1 Failure-Weighted Pathwise Audit Plan

## Objective

Test one mechanism suggested by the valid E008 audit without changing the
tracking task: whether production's uniform mean of norm-clipped per-environment
pathwise gradients underweights the worst carried fragments.

No training is authorized by this plan.  First construct an equal-functional
one-step candidate and require broad, safe ordinary transfer.

## Frozen control

- E064 final actor, normalizer, carried states, reference, controller, plant,
  reward, termination, solver 4/5, horizon 48, population 64, sigma 0.1,
  gamma 0.99, shard seeds 0/1/2/3, and held-out seeds 4/5/6/7.
- Production finite sanitization and per-environment norm-1 clipping.
- Baseline, candidate functional RMS 0.01, and contemporaneous fused ordinary
  phases 0/100/200/300/400.
- E066 baseline competence floors 110/78/74/76/58.

## Single causal change

Within each shard, bin the frozen initial phases into five fixed 100-frame
bins.  Rank the stop-gradient per-environment negative-return losses within
each nonempty bin, breaking ties by environment index.  Select the worst
`max(1, floor(n_bin / 4))` fragments.

Mix half production-uniform mass and half selected-tail mass while preserving
each phase bin's original total probability:

`weight_i = 0.5 / 64 + 0.5 * (n_bin / 64) * selected_i / k_bin`.

Apply these weights only to the already clipped actor gradients.  Rewards,
individual losses, dynamics, phases, and critic behavior remain unchanged.
Weight/rank receipts may use float64, but every weighted aggregate leaf and
candidate parameter leaf must retain the exact corresponding production
uniform/baseline dtype.  Candidate output separation is descriptive telemetry,
not an outcome gate.

## Implementation tasks

1. Add pure phase-bin ranking, exact weight construction, weighted PyTree
   aggregation, stability summaries, and fail-closed validation.
2. Add a three-candidate classifier for baseline, production-uniform pathwise,
   and failure-weighted pathwise.
3. Reuse the existing E064 prepared execution, candidate calibration,
   held-out summaries, ordinary phase grid, receipts, and atomic publication;
   avoid copying the existing transaction when a narrow reusable seam exists.
4. Add adversarial unit tests before implementation: exact phase masses,
   inclusive small bins, deterministic ties, malformed inputs, independent
   recomputation, stability, phase safety, and two-phase breadth.
5. Freeze a clean code commit, preregister the experiment, validate/dry-run,
   then execute once.  If positive, require an independent process repeat
   before any training continuation because E007/E008 showed only 0.835
   cosine between nominally identical aggregate clipped pathwise gradients.

The validity receipt keeps exact uniform reproduction, exact failure-weighted
reproduction, and exact stability-geometry binding as three separate gates.
An independent NumPy host reconstruction is additionally required within 256
frozen dtype-scaled tolerance units; exact producer replay remains mandatory.

## Decision gates

The candidate is supported only if:

- every weight/rank/gradient receipt recomputes exactly and all evidence is
  finite, fresh, complete, and replay-free;
- its within-run shard stability is no worse than uniform pathwise by more
  than 0.05 in pairwise mean/minimum and leave-one-shard-out minimum cosine;
- its mean reward is at least `max(baseline, uniform) + 0.001` or its mean
  survival is at least `max(baseline, uniform) + 5`;
- every phase retains at least 90% of baseline survival; and
- at least two phases each improve over uniform by reward at least 0.001 or
  survival at least 5.

Otherwise classify the fixed weighting as not material or unstable and do not
tune its quartile or mixture coefficient.
