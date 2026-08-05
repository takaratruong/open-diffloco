# G1 Fixed-Mass Headroom Diagnostic Design

## Objective

Explain why the residual trained through a fixed 1.15 body-mass-and-inertia
model improves every paired phase but gains only `0.000197` aggregate reward.
The discriminator must separate insufficient task headroom from an
update-limited residual without changing the policy, checkpoint, model, reward,
optimizer, or shift magnitude.

## Design

Extend the existing paired residual evaluator with an optional nominal source
baseline. In one process and with one loaded source policy, evaluate:

1. source controller in the nominal `1.0` mass model;
2. source controller in the shifted `1.15` mass model;
3. fixed step-64,512 residual in the shifted `1.15` mass model.

Use phases `0`, `30`, `60`, and `90`, strict terminations, seed zero, solver
4/5, and every remaining E-20260805-085 setting. The JSON reports the three
aggregates, shifted-source minus nominal-source deltas, and shifted-residual
minus shifted-source deltas. Same-process execution avoids using separate-run
contact variability as a causal estimate.

During each shifted residual step, retain the exact bounded correction before
adding it to the source action. Across all phase suffixes report mean absolute,
root-mean-square, maximum absolute, and fraction of action elements at or above
95 percent of the `0.1` residual bound. This distinguishes a near-zero learned
correction from a saturated correction whose objective does not translate into
reward.

## Outcomes

- `insufficient-shift-headroom`: shifted source adds no terminal and loses less
  than `0.001` aggregate reward relative to nominal source. Select a more
  discriminative, physically justified plant shift rather than tuning the
  optimizer.
- `material-headroom-recovered`: shifted source loses at least `0.001` reward
  or adds a terminal, while the shifted residual recovers at least `0.001`
  reward without an additional terminal. The adaptation mechanism passes and
  can inform collocation transfer after visible confirmation.
- `material-headroom-update-limited`: shifted source has material degradation,
  but the shifted residual recovers less than `0.001` reward. Localize the
  residual objective/update using the action-magnitude evidence.
- `invalid-execution`: any fixed checkpoint, model, phase, or complete JSON
  evidence is missing, inconsistent, non-finite, or timed out.

The `0.001` threshold matches E-20260805-085 and is fixed before execution.
No sweep or post-hoc threshold change is permitted.

## Implementation And Tests

Add `--baseline-body-mass-scale` to
`tools/compare_g1_tracking_residual.py`. When absent, preserve the existing
two-controller document. When present, add baseline-source phase and aggregate
records plus shifted-minus-baseline deltas. Add a pure
`summarize_action_deltas(deltas, bound)` function with finite, positive-bound,
and shape validation.

Unit tests cover exact summary values, empty/nonfinite/invalid-bound rejection,
CLI transport, and unchanged ordinary rollout reduction. Run the full residual
comparison, evaluator, and runner test modules before pinning the executable
commit.

The diagnostic is preregistered and executed once through `tools.runexp`.
Canonical result, artifact hash, hypothesis edge, finding, state, and Obsidian
projection are published before selecting the next mechanism.
