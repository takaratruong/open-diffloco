# Low-Actor-LR Early-Learning Gate Design

## Decision

Run one fresh, seed-zero successor to E-20260813-004 that changes only the
actor learning rate from `5e-3` to `1e-3`. Repeat the exact 98,304-transition
bounded-mean early-learning gate and its checkpoint-bound noisy and clean
rollouts. Do not add learning-rate warmup, decay, action clipping, reset noise,
observation noise, domain randomization, pushes, assistance, or another
optimizer/objective change.

## Scientific Question

Does a fivefold lower actor learning rate prevent the rapid tanh-boundary
crowding seen in E-20260813-004 while retaining at least its clean phase-zero
survival?

E004 completed 16 valid effective-512 H12 updates, but at step 98,304 its
actor-mean RMS was `0.8345`, maximum magnitude was `0.9996`, and `53.22%` of
mean coordinates had `|mean| >= 0.95`. Its first stochastic episode survived
16 transitions and its clean phase-zero rollout survived 40. This was not an
initialization property: E003 had zero saturated mean coordinates after one
update, and the separate E000 stream rose from zero mean saturation at 98,304
to `66.08%` at 786,432. The next uncertainty is therefore optimizer step size,
not action-noise plumbing.

## Alternatives Considered

1. **Fixed actor LR `1e-3` — selected.** It is a strong, interpretable
   fivefold reduction and matches the working Open-DiffLoco-scale rate more
   closely. One scalar changes.
2. **Fixed actor LR `2.5e-3`.** This preserves more aggressive learning but may
   be too weak a discriminator after E004 reached `53.22%` saturation in only
   16 updates.
3. **Warmup or learning-rate decay.** This could retain a larger later rate,
   but it introduces schedule shape and duration as additional causal factors.
   It is deferred unless the fixed lower rate prevents saturation but clearly
   underlearns.

## Frozen Treatment

Inherit the complete E004 treatment and evidence contract:

- fresh zero-output actor, seed zero;
- `g1_tracking_rmr_50hz_decoupled_exploration`;
- tanh-bounded actor mean and unclipped reparameterized sampled action;
- scalar `1.0` to exact 29-joint learned RMR standard deviation over 800,000
  transitions;
- full reference-residual scale `1.0`;
- clean observations and exact reference-state resets;
- nominal effort, gains, mass, friction, and flat terrain;
- no pushes, reset perturbations, domain randomization, or assistance;
- future reference lookaheads `4/8/12`;
- 256 physical environments with two-way gradient accumulation, H12, five-bin
  CAGrad, the existing critic settings, and `g1-4x5` solver;
- exactly 98,304 transitions / 16 actor updates and a final step-98,304
  checkpoint;
- exact phase-zero 120-transition stochastic training-distribution replay with
  ordinary auto-resets, plus a deterministic clean phase-zero rollout.

The sole treatment delta is:

```text
actor_lr: 0.005 -> 0.001
```

Critic learning rate and every non-actor optimizer parameter remain unchanged.

## Implementation

Extend `tools/run_g1_rmr_action_space_parity.py` with an explicit low-actor-LR
early-learning mode layered on `build_decoupled_early_learning_kwargs`. The new
builder must differ from that parent builder only in `actor_lr`. Use a distinct
CLI flag rather than silently changing E004 behavior. Require the flag only
with `--early-learning-gate --decoupled-exploration`, and fail closed on any
other combination.

Generalize the existing early-learning hparam validator to accept an explicit
expected actor LR supplied by the selected mode. Legacy E004 validation must
continue to require `5e-3`; the successor must require `1e-3`. Preserve the
existing action-tape equation checks, trajectory reconstruction, media gates,
diagnostic cadence, checkpoint telemetry, provenance, and manifest-last
validation behavior.

## Evaluation And Outcomes

The primary comparison is the exact E004 endpoint:

| Metric | E004 | Low-LR advancement gate |
|---|---:|---:|
| Actor-mean saturation, `|mean| >= 0.95` | `0.5322` | `< 0.20` |
| Actor-mean maximum | `0.9996` | `<= 1.000001` |
| Clean phase-zero survival | `40` | `>= 40` |
| Noisy first-episode survival | `16` | descriptive, must be finite |

All 16 updates must also remain finite, all five final CAGrad bins must be
occupied and finite, and diagnostic actor gradient/update norms must be
positive and finite.

The preregistered outcomes are:

1. **Lower LR prevents saturation without regression.** Saturation is below
   `0.20`, clean survival is at least 40, and all execution gates pass. This
   authorizes a separately registered longer `1e-3` continuation.
2. **Lower LR prevents saturation but underlearns.** Saturation is below
   `0.20`, but clean survival is below 40. Do not extend; next test a moderate
   fixed LR such as `2.5e-3`, not a new stochastic treatment.
3. **Lower LR does not prevent saturation.** Saturation remains at least
   `0.20`. Reject actor LR as a sufficient fix and move to a preregistered
   trust-region or explicit mean-margin objective.
4. **Invalid execution.** Any code, treatment, provenance, checkpoint,
   telemetry, action-tape, trajectory, or media contract fails. Repair only
   the execution defect through a separately pinned successor.

The run stops after the 98,304-transition gate regardless of outcome. It does
not automatically continue training.

## Verification

Tests must prove:

- the low-LR builder changes only `actor_lr` relative to E004;
- invalid CLI mode combinations and nonzero seed fail before training;
- E004 and low-LR validators require their respective exact rates;
- execute selects the low-LR builder and writes the same gated evidence only
  after validation;
- all existing parity, E003, and E004 runner/evaluator tests remain green.

Before GPU use, run focused tests, Ruff, `py_compile`, `git diff --check`, and
an independent read-only launch review. Register the experiment in the lab at
the final clean code commit, validate the registry, and dry-run `runexp` before
the sole guarded execution.
