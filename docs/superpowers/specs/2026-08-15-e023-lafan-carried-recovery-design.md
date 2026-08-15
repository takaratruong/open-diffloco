# E023 LAFAN Carried-Recovery Design

## Objective

Test whether the E026 mechanism generalizes to a new reference: freeze the
better zero-shot E023 actor and train a new zero-effect residual adapter only
from E023's own pre-failure states on the pinned 500-state LAFAN reference.
This is a one-seed mechanism test, not a repeatability or sim-to-real claim.

## Evidence and Alternatives

On the exact LAFAN phase grid 0/100/200/300/400, E023 survives
118/63/49/39/46 transitions. E026 survives 106/61/48/40/46, so its learned
short-walk adapter does not transfer. Continuing that adapter is rejected.
Restarting a full actor is also rejected because it discards E023's useful
basin. The treatment therefore transfers only E026's architecture.

## Treatment

- Resume the exact E023 checkpoint at step 1,572,864.
- Explicitly authorize changing only the reference path from the short walk to
  the pinned 500-state LAFAN reference; preserve stride one and every other
  actor, optimizer, plant, reward, termination, solver, and noise contract.
- Freeze E023 actor parameters, parent Adam moments, and actor normalizer.
- Add a fresh 328-256-29 ELU residual adapter initialized to exact zero with
  zero Adam moments.
- Collect 24 history-faithful nonterminal states, 6-29 transitions before
  failure, from each exact start 0/100/200/300/400: 120 rows total.
- Use 25-percent carried-bank resets and 75-percent exact uniform LAFAN RSI.
- Preserve H24, effective population 512, five-bin CAGrad, per-environment
  gradient clipping, zero terminal bootstrap, nominal physics, and the exact
  RMR endpoint action-noise vector.
- Use clean-action anchor weight 1.0 and train at most 64 adapter updates, with
  immutable checkpoints every 8 updates.

## Migration and Provenance

The reference-path change must fail closed by default. A new explicit resume
authority is valid only when a real checkpoint is resumed and both the parent
and requested reference paths are present, distinct, absolute files. Persist
the parent/requested paths and hashes in a migration report and hparams. The
residual migration must remain exactly action-equivalent before the first
update. The bank binds the E023 checkpoint/hparams, LAFAN reference, model,
controller, solver, code commit, source phases, observed survival, and every
array hash.

## Evaluation and Selection

Evaluate adapter updates 8/16/32/64 replay-free at phases
0/100/200/300/400. Eligibility requires componentwise preservation of the
E023 baseline 118/63/49/39/46. Select eligible checkpoints lexicographically by
minimum, median, mean survival, then earlier update. A clean solve additionally
requires every suffix to reach its exact remaining transition count with no
true terminal. Render the selected phase-zero clean and training-noise
episodes and publish survival, adapter-gradient/update, anchor, and CAGrad
curves.

## Outcome Map

1. `lafan-carried-solves`: all five suffixes complete terminal-free.
2. `lafan-carried-advances`: a checkpoint preserves every E023 phase and
   improves at least one phase.
3. `lafan-carried-insufficient`: no checkpoint preserves E023 componentwise.
4. `invalid-execution`: any reference migration, bank, frozen-state, finite
   gradient, CAGrad, checkpoint, evaluator, provenance, or media gate fails.

## Tests

- Reference-path changes reject by default, reject missing resume metadata,
  and pass only with explicit authority and exact paths/hashes.
- Legacy resumes remain byte-behavior compatible.
- Bank validation requires exactly five 24-row bands with exact history and
  preterminal distances.
- Runner tests pin the single causal delta and all eight checkpoints.
- Selection rejects compensating phase regressions and never calls a terminal
  final transition a solved suffix.

