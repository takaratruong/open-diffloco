# Conflict-Projected Recovery-Teacher SHAC Design

## Objective

Test whether E036's demonstrated recovery actions can safely expand E023 when
the teacher signal is applied explicitly during differentiable-physics policy
learning. E041 established that transferring E038's hidden representation alone
is insufficient: its closest checkpoint improves fourteen carried states but
still regresses one.

## Alternatives considered

1. **Weighted imitation plus SHAC loss.** This is the smallest implementation,
   but a scalar loss weight is not comparable to the five-bin CAGrad gradient
   scale and can silently dominate or disappear as training changes.
2. **Treat imitation as a sixth CAGrad task.** This exposes the full
   multi-objective geometry, but changes the registered five-bin CAGrad contract,
   telemetry, reducer, and selection machinery for a single discriminator.
3. **Conflict-projected, norm-capped teacher gradient (selected).** Compute the
   existing five-bin CAGrad direction first, remove only the teacher component
   that opposes it, cap the remaining teacher contribution relative to the
   physics direction, and apply one shared optimizer update. This keeps the
   existing phase optimizer intact while making teacher influence explicit and
   bounded.

## Fixed treatment

- Resume the exact E023 TrainState, reference migration, carried-state bank,
  and zero-head E038 hidden-feature initialization used by E041.
- Use exactly the 416 successful E036 transitions selected by its immutable
  non-contiguous success mask. Validate the dataset SHA-256, tensor names,
  shapes, finite values, parent-plus-correction identity, and effective-action
  boundary before compilation.
- Normalize the stored 10-by-328 actor histories with E023's frozen actor
  normalizer. Apply the current residual adapter to the newest normalized frame.
- Use E038's exact teacher objective:

  `mean((predicted_correction - teacher_correction)^2)` plus
  `mean((clip(parent_action + predicted_correction, -1, 1) - teacher_effective_action)^2)`.

- Differentiate the full-batch teacher objective once per SHAC actor update.
  The E023 parent parameters and actor normalizer remain frozen exactly.
- Let `p` be the existing five-bin CAGrad actor gradient and `t` the teacher
  gradient. If `dot(p, t) < 0`, replace `t` with
  `t - dot(p,t) / (||p||^2 + 1e-12) * p`; otherwise retain `t`.
- Scale the projected teacher gradient by
  `min(1, 0.5 * ||p|| / (||t|| + 1e-12))`, so its applied norm is at most half
  the physics-gradient norm. Apply `p + scaled_t` through E041's unchanged
  adapter-only optimizer and global update boundary.
- Persist teacher loss, raw/projected/applied norms, pre-projection cosine and
  dot product, applied scale, final combined norm, finite validity, and exact
  zero parent-gradient/update/moment drift at every diagnostic checkpoint.
- Preserve E041's H24, effective population 512, five phase bins, CAGrad
  settings, per-environment clipping, 25-percent history-faithful carried
  resets, RMR action noise, zero bootstrap, parent-action anchor 1.0, nominal
  physics, action boundary, solver, seed zero, 64-update budget, and immutable
  checkpoints after updates 8/16/32/64.

## Resume and provenance boundary

Teacher regularization defaults off. Enabling or changing its dataset path,
dataset SHA-256, gradient ratio, or projection rule on resume requires explicit
authority. A missing or partial resumed hparams record fails closed. Persist the
exact dataset and treatment metadata in `hparams.json` and a manifest-last
teacher preflight report. The treatment may not start if E041's zero-head
migration or the E036 dataset contract fails.

## Selection

Use E041's repaired paired-parent evaluator and selector. Evaluate updates
8/16/32/64 replay-free on the ordinary five-phase LAFAN grid and all 120 H32
carried states. Each checkpoint is compared only to the E023 parent arm in the
same hash-bound paired execution because separate MJX GPU processes are not
bitwise deterministic. Eligibility still requires componentwise non-regression
on all 120 paired carried survivals. Select lexicographically by phase-zero H32
count, total H32 count, minimum/median/mean carried survival, ordinary
minimum/median/mean, then earliest update.

## Outcome map

- `teacher-objective-solve`: an eligible checkpoint completes every ordinary
  suffix and all 120 H32 carried recoveries.
- `teacher-objective-advance`: an eligible checkpoint improves at least one
  carried state or ordinary phase without any paired carried regression.
- `teacher-objective-insufficient`: execution is valid but no checkpoint is
  eligible or no eligible checkpoint improves E023.
- `invalid-execution`: any provenance, dataset, zero-head migration,
  frozen-state, teacher-gradient, CAGrad, finite-value, action-boundary,
  checkpoint, evaluation, or artifact gate fails.

## Testing and evidence

- Pure tests cover dataset validation, exact loss reproduction, tree dot/norm,
  conflict projection, norm capping, zero-physics behavior, nonfinite rejection,
  resume authority, and scalar telemetry.
- Integration tests cover one compiled two-shard actor update with a nonzero
  teacher gradient, exact parent preservation, disabled-path equivalence, and
  checkpoint/hparams persistence.
- The guarded runner validates the E041 migration, all eight training archives,
  finite teacher and CAGrad telemetry, and the four registered evaluation
  checkpoints before any scientific classification.

## Limits

This is one fixed teacher-gradient ratio and one fixed motion/bank. It does not
test a weight sweep, teacher dataset expansion, online relabeling, new action
authority, or sim-to-real transfer. A negative result rejects this bounded
explicit-teacher recipe, not all demonstration-assisted differentiable physics.
