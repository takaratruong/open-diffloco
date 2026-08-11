# G1 Effective-512 Early Gate Design

## Objective

Test whether doubling the number of independent differentiable trajectories
averaged into each canonical G1 SHAC actor update improves replay-free
long-motion tracking at the same 128-update early gate.

## Design

Directly differentiating 512 G1 environments exceeds one L40S's memory. Use
the already verified sequential-accumulation path instead: two physical
256-environment shards are generated independently, differentiated at the
validated physical width, averaged, and followed by one actor update. This is
an effective batch of 512 without changing the optimizer-update count.

Create one immutable runner derived from `build_canonical_kwargs`. Relative to
the canonical contract it changes only:

- `gradient_accumulation_steps`: `1` to `2`;
- `total_steps`: `393,216` to `786,432`, the mechanically required budget for
  128 updates of `512 * 12 = 6,144` transitions each.

Keep the physical `num_envs=256`, horizon 12, fixed 4x5 solver, seed, model,
reference, actor, critic, learning rates, noise, randomization, curriculum,
rewards, observations, and action semantics unchanged. The runner exposes no
scientific overrides.

## Evidence Gate

Require finite completion, complete artifacts, exact effective-batch
provenance, and nominal replay-free evaluation from exact phases
0/100/200/300/400. Compare against the canonical 128-update vector
14/43/47/61/19. Advance to a separately registered 512-update continuation
only when minimum survival is at least 19 and median survival is at least 48.
Otherwise stop effective-batch scaling for this canonical formulation.

## Limits

This is a seed-zero early gate. Passing would justify longer training, not
establish repeatability, complete suffix tracking, transfer, or sim-to-real
readiness. Failing would reject this batch-size intervention, not SHAC or
differentiable-physics training generally.
