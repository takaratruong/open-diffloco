# G1 Motion-Anchor Position Full-Budget Design

## Decision

Run one fresh 128-update H24 SHAC experiment using the exact successful E004
`motion_anchor_pos_b` observation and otherwise the complete E023 recipe. E004
already proves an early componentwise improvement at updates 16 and 32. This
successor tests whether that advantage persists at the full E023 budget.

Keep E004 immutable. Extend its runner with an explicit `early`/`full` budget
contract whose default remains `early`, so the executed E004 command and tests
remain unchanged. E005 passes `--budget full`.

## Alternatives Considered

1. **Recommended: fresh full-budget run through the same runner.** This gives a
   matched seed-zero comparison with E023 and reuses the reviewed preflight,
   trainer, checkpoint validator, and observation contract with little code.
2. Resume E004 update 32. This saves some transitions but changes the execution
   and provenance boundary, needs resume authority/migration evidence, and is
   less direct than a fresh matched E023 control.
3. Add a separate copied full-budget runner. This isolates commands but
   duplicates preflight and artifact-validation logic that has already caused
   real gaps. Reject it.

## Scientific Question

At a matched 128-update budget, does adding only body-frame motion-anchor
position error preserve or improve E023's replay-free five-phase walking
competence?

## Causal Contract

The only learning-semantic difference from E023 is
`actor_observe_motion_anchor_position=true`. Preserve:

- ten-frame history and 331-value treatment frames, for a 3,310-value input;
- H24, 256 physical environments, two accumulation shards, and effective
  population 512;
- five-bin CAGrad, per-environment clipping, actor/critic architecture and
  learning rates;
- exact reference resets, nominal physics, zero pushes, no observation noise,
  zero terminal bootstrap;
- full reference-residual action authority and the action-parity boundary;
- RMR action noise ending at transition `1,572,864`;
- seed zero, reference, model, controller, solver, and x64 execution.

The full budget is `1,572,864` transitions / 128 updates. Checkpoints remain at
every 16 updates, producing exact archives at updates
`16/32/48/64/80/96/112/128`.

## Evaluation

First require deterministic early corroboration at updates 16 and 32. Their
five-phase vectors must exactly reproduce E004's `43/38/48/49/24` and
`51/66/55/49/24`; otherwise classify invalid execution before interpreting
later learning.

Evaluate updates 64 and 128 replay-free at phases `0/25/50/75/100`. Matched
E023 controls are:

- update 64: `71/60/51/49/24`;
- update 128: `116/99/67/49/24`.

Update 64 is a registered learning checkpoint. Update 128 is the decisive
full-budget comparison and the only checkpoint eligible to replace E023.
Render update 128 from phase zero and publish its trajectory, summary, MP4,
contact sheet, learning/action plot, comparison, and manifest-last evidence.

## Outcomes

- `root-position-full-advances`: update 128 preserves all five E023 update-128
  survival values and strictly improves at least one of phases `0/25/50/75`.
- `root-position-full-parity`: no advance, but update 128 is within two
  transitions of E023 update 128 at every phase.
- `root-position-full-mixed`: neither advance nor parity, and update 128 gains
  more than two transitions at one phase while losing more than two at another.
- `root-position-full-underperforms`: valid execution satisfies none of the
  preceding outcomes.
- `invalid-execution`: provenance, single-delta, early-correspondence,
  checkpoint, finite-gradient, CAGrad, evaluator, media, or artifact validation
  fails.

An advance retains the E005 update-128 actor and promotes a reusable finding
that root-translation observability improves differentiable-physics tracking.
Parity retains E023 while keeping the feature optional. Mixed or underperforming
results reject the feature as a full-budget replacement even though it helped
early learning.

## Implementation Boundary

Generalize only the existing E004 runner's execution-budget contract. Do not
change the environment, trainer, actor, evaluator, objective, or simulation.
Test first that:

- omitted budget remains byte-for-byte E004 early semantics;
- full mode changes only `total_steps` relative to E004 and changes no
  scientific kwarg relative to E023 besides the observation flag/input guard;
- the full preflight records all eight updates/steps and remains JAX-free;
- artifact validation requires exactly eight finite archives;
- the full classifier covers every registered outcome and rejects malformed or
  noncorroborating evidence.

## Stop Rule

Stop after update 128 and curate the registered outcome. Do not add features,
resume farther, tune hyperparameters, or launch another run before the full
phase-grid and media evidence are validated.
