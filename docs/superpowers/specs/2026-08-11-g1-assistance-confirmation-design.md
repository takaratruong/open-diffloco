# G1 Assistance-Curriculum Independent-Randomness Confirmation

## Purpose

E012 showed that a continuously vanishing analytic torso wrench can improve an
unassisted G1 policy: the selected checkpoint raised the minimum five-phase
survival from E008's 44 to 51 after eight fully unassisted updates. The next
uncertainty is whether that improvement is repeatable under an independent
training-randomness stream or was specific to E012's restored checkpoint RNG.

This experiment changes only post-resume stochastic evolution. It is not a
schedule sweep, a second assistance cycle, or a new objective.

## Causal boundary

Resume the exact selected E008 checkpoint at step 1,327,104. Preserve its:

- actor, critic, target critic, and optimizer state;
- actor and critic normalizers;
- physical environment state, reference phases, history, actions, and domain
  randomization values at the resume boundary;
- H12 effective-512 five-bin CAGrad objective;
- continuous assistance schedule, zero-assistance cohort, solver, reward,
  termination, reset noise, domain randomization, and checkpoint cadence.

Replace only:

1. `TrainState.key`, using a namespaced key derived from confirmation seed 1;
2. every per-environment `env_state.info["rng"]` key, using a different
   namespace derived from the same confirmation seed.

Using the existing `--seed` argument alone is explicitly insufficient because
exact resume restores both of these saved RNG sources. The implementation must
therefore expose an opt-in `resume_random_seed: int | None = None` boundary.
The default remains `None`, preserving all existing exact-resume behavior.

## Rekey contract

A pure helper accepts a resumed `TrainState` and a non-negative integer seed.
It derives disjoint trainer and environment roots by `jax.random.fold_in` with
fixed namespace constants. It replaces the trainer key and splits the
environment root into exactly the existing number of per-environment keys.

The helper must fail closed when:

- the seed is a boolean, negative, or non-integer;
- `env_state.info` has no `rng` leaf;
- the saved RNG leaf is not rank two with trailing width two;
- rekeying is requested without a resumed checkpoint.

An audit artifact records the seed, namespaces, old/new key hashes, changed-key
booleans, and exact zero drift for every non-RNG TrainState leaf. It must not
serialize raw random keys as scientific output.

## Immutable confirmation runner

Add `tools/run_g1_frozen_residual_assistance_confirmation.py`. It reuses the
E012 runner kwargs exactly and changes only `resume_random_seed=1`. Its normal
`seed` remains zero so unused fresh-initialization paths cannot become an
unregistered second change. It resumes E008, not E012, and repeats the same
64-update schedule through step 1,720,320.

The runner must persist `resume_random_seed=1` and the rekey audit in the
training directory. Existing E012 artifacts and runners remain immutable.

## Evaluation and decision

Evaluate all eight checkpoints with exactly zero external wrench at phases
0/100/200/300/400. Select by minimum survival, then median, mean, and earliest
step, exactly as E012.

The confirmation passes if the selected key is lexicographically greater than
E008's `44/70/68.4`. E012's `51/66/68.6` is reported as a descriptive
comparison, not the confirmation threshold. This avoids requiring an
independent repeat to reproduce one seed's exact magnitude.

- **Repeatable advance:** finite valid execution and selected minimum-first key
  above E008. Retain the curriculum and proceed to a separately registered
  full-suffix intervention from the best validated checkpoint.
- **Finite non-confirmation:** valid execution but no selected key above E008.
  Treat the curriculum gain as seed-sensitive and redesign it before stacking
  another cycle.
- **Invalid execution:** any provenance, rekey, preservation, update-count,
  finiteness, assistance, CAGrad, evaluator, or artifact gate fails. Repair only
  the execution defect.

No claim of complete tracking, transfer, hardware readiness, or sim-to-real
success follows from this confirmation alone.

## Verification

Focused tests must prove:

- default `None` preserves a resumed state exactly;
- seed 1 deterministically changes both RNG sources;
- different confirmation seeds produce different keys;
- every non-RNG leaf remains exact;
- invalid seeds and malformed/missing environment RNG leaves fail closed;
- the immutable runner differs from E012 only by the rekey setting;
- existing E012 runner and assistance tests remain green.

The registered first completed update is the operational GPU gate. It must have
finite actor gradients, valid CAGrad/preview/wrench telemetry, five occupied
phase bins, and a plausible assistance active fraction before the run may be
interpreted scientifically.
