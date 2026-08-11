# G1 Preview-Preservation Ladder Design

## Objective

Exploit the verified later-phase benefit of future reference without erasing
the byte-exact E008 policy's existing phase competence. The ladder addresses:

1. full-network interference;
2. redundant future-reference history;
3. mismatch between the differentiable actor objective and replay-free
   phase-grid survival.

The target remains closed-loop G1 tracking trained through differentiable MJX
physics. No PPO teacher, replay controller, hard-coded phase switch, preview
offset sweep, longer horizon, or hardware claim is introduced.

## Evidence Boundary

E008 survives `28/55/77/21/45` frames from phases
`0/100/200/300/400`. E010's selected midpoint survives
`19/58/65/55/45`, and its final checkpoint survives
`18/80/78/50/44`. The append-only E010 migration begins action-equivalent to
E008, so future information is useful at later phases, but unrestricted actor
and normalizer updates redistribute competence and regress phase zero.

CAGrad is conflict-aware rather than a non-regression constraint. It can find
a finite compromise between phase-conditioned training gradients without
preserving closed-loop survival at every phase.

## Considered Architectures

### A. Masked current-preview columns — selected first

Reuse the proven 3,280-value E010 observation and migration, but make only the
newest history frame's 174 preview-to-first-hidden-layer kernel rows trainable.
All legacy actor parameters, all preview rows belonging to the older nine
history frames, biases, later layers, and actor-normalizer statistics remain
frozen at their migrated E008 values.

This leaves only `174 * 512 = 89,088` trainable actor values. Although the
environment still carries the older preview fields for checkpoint/layout
compatibility, their first-layer rows remain exact zero and therefore cannot
affect the action. The actor is functionally a current-time preview adapter
inserted into E008's learned latent space.

Advantages:

- minimal change to a path already proven executable;
- exact E008 action at initialization;
- legacy mapping and normalization cannot drift;
- isolates whether current preview alone can improve the parent;
- self-contained checkpoints and existing evaluator compatibility.

Limitations:

- the adapter can still produce a harmful correction;
- the 3,280-value physical tensor remains computationally larger than needed;
- frozen downstream features may limit how preview information is expressed.

### B. Separate preview residual MLP

Freeze E008 and learn a small preview-conditioned action residual. This gives
the cleanest conceptual boundary and could enforce a smooth action cap, but it
requires new actor composition, optimizer, migration, checkpoint, evaluator,
and normalization paths. It is deferred unless masked columns lack expressive
power.

### C. Unfrozen actor with parent-policy trust

Penalize deviation from E008 on phase-balanced states while continuing to
train the full actor. This retains maximum capacity but does not structurally
prevent internal forgetting and introduces a trust coefficient. It is not the
first experiment.

## Stage 1: Frozen-Parent Current-Preview Adapter

Resume the exact E008 checkpoint, not E010. Apply the already validated
append-only `(4, 8, 12)` migration. Before optimization:

- legacy actor rows and non-input actor leaves equal E008 exactly;
- all new preview rows and their Adam moments equal zero;
- the complete migrated actor agrees with E008 within `1e-7`;
- actor-normalizer legacy statistics are exact and preview statistics are the
  deterministic reference statistics already used by E010.

For every update, reduce the ordinary five-bin CAGrad gradient and then apply
an elementwise trainable mask. Exactly the newest frame's preview rows in
`params/Dense_0/kernel` survive the mask; every other actor-gradient value is
zero before global clipping and Adam. Because E008 has nonzero Adam moments,
zero gradient alone is insufficient: mask the resulting parameter updates and
merge the new optimizer state so that frozen `mu`/`nu` entries remain bit-exact
while the global Adam count and authorized preview entries advance. Freeze the
complete actor normalizer after migration. Critic parameters and critic
normalizer remain trainable as in E008.

The run retains E008's H12, 256 physical environments, accumulation factor
two, five CAGrad bins, 4x5 solver, randomization, seed zero, and 393,216
additional-transition budget. Save midpoint and final checkpoints.

### Stage 1 diagnostics

Each diagnostic record must include:

- trainable preview-gradient norm and preview-update norm;
- maximum absolute update outside the authorized rows, which must be zero;
- maximum absolute drift of every frozen actor parameter, frozen Adam moment,
  and actor-normalizer value from the migrated parent, which must be zero;
- per-phase-bin mean and maximum absolute action deviation from the parent;
- ordinary finite CAGrad telemetry and complete phase-bin counts.

The parent action is computed before action noise by applying the current actor
to the same noisy normalized legacy observation after zeroing the newest
preview slice. Because all non-preview parameters are frozen, this is exactly
the migrated E008 actor under the same observation corruption and does not
require a second parameter copy inside the compiled update.

### Stage 1 decision

Evaluate midpoint and final at phases `0,100,200,300,400` with the existing
nominal replay-free protocol. Select by maximum minimum survival, then median,
mean, and earliest checkpoint.

- Advance if minimum survival is at least 21, median at least 48, and every
  phase is no more than five frames below E008 (`23/50/72/16/40`).
- If execution is finite but the selected policy regresses a parent floor
  while improving at least one later phase, proceed to Stage 2.
- If preview updates are nonzero but no phase improves over E008, stop the
  masked-column architecture and design the separate residual MLP.
- Any mask leak, normalizer drift, invalid telemetry, or provenance mismatch
  is invalid execution and authorizes only a repair.

## Stage 2: Phase-Balanced Parent-Action Trust

Stage 2 is conditional and separately preregistered. It resumes exact E008 and
uses the same Stage-1 masked adapter. Add only a parent-action trust mechanism.
Do not tune preview offsets or unfreeze the parent.

Use Stage 1's phase-binned action-deviation telemetry to pin one trust scale
before execution. The preferred mechanism is a smooth bound on the adapter's
action displacement relative to the zero-preview parent; if the observed
action geometry makes such a cap inappropriate, use one fixed phase-balanced
mean-squared parent-action penalty. The choice and numerical value must be
written before the run, not selected from Stage-2 results.

The trust term is measured in each of the same five phase bins before CAGrad
combination, preventing densely sampled phases from diluting protection for a
weak phase. Stage 2 uses the same training and evaluation budget as Stage 1.

Advance only under the same aggregate and per-phase gates. If Stage 2 cannot
retain the parent floors, stop the preview-column family rather than tuning
the trust value repeatedly.

## Objective–Survival Audit

At the exact midpoint and final updates, training persists a
`checkpoint_phase_metrics.json` record containing the differentiable actor
loss and parent-action deviation for each of the same five CAGrad bins, plus
the global preview update norm. After both replay-free grids finish, a pure
evaluator combines those immutable training records with phase survival and
writes `objective_survival_audit.json`.

The audit pairs the bin containing each registered evaluation start with that
phase's replay-free survival, producing ten checkpoint-phase cases. It is a
bounded audit, not a new optimizer. Report rank/order agreement between actor
loss and survival. If the training objective improves while survival
consistently worsens, the next architecture must introduce a rollout-level
constraint or selection signal; do not continue adjusting observation
encodings.

## Verification

Unit tests must establish RED then GREEN for:

- the exact single-slice gradient mask;
- zero frozen-parameter and frozen-moment updates over repeated synthetic Adam
  steps with nonzero inherited moments;
- frozen actor-normalizer behavior;
- exact zero-preview parent-action reconstruction;
- phase-binned action-deviation telemetry;
- immutable runner arguments and evaluator propagation;
- exact E008 migration and pre-update action equivalence.

Run one focused scientific suite before registration. Each GPU experiment is
registered, dry-run, executed alone, evaluated deterministically, and curated
before the conditional successor is designed or launched.

## Stop Conditions

- No preview-offset, learning-rate, horizon, seed, or trust-value sweep.
- No more than Stage 1 plus one evidence-pinned Stage 2 run in this ladder.
- No hard-coded phase gate or policy switch based on the five evaluation
  phases.
- If both runs miss parent non-regression, preserve E008 and move to a
  separately designed residual architecture or rollout-level objective.
