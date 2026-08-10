# G1 Fixed-Horizon-24 Early-Gate Design

## Objective

Test whether exposing the G1 SHAC actor to 24 differentiable simulator steps,
instead of 12, improves robust replay-free tracking at the same early training
budget. The experiment answers one causal question and must not become another
unbounded training campaign.

## Scientific Contract

The successor starts from the canonical seed-0 initial state and changes only
`unroll_length` from `12` to `24`. It preserves the canonical G1 model,
50 Hz reference, 256 environments, fixed-scan `g1-4x5` solver, actor and critic
architectures, learning rates, terminal critic, rewards, residual-action
semantics, observation history, noise, domain randomization, curriculum, and
checkpoint interval.

The authorized budget is exactly 393,216 transitions. This is the existing
horizon-12 checkpoint budget, not a second scientific change. The runner must
finish normally and emit final training artifacts at that budget; it must not
resume from E003 or accept scientific CLI overrides.

## Runner Boundary

Add a dedicated `tools/run_g1_horizon24_shac.py` runner. It imports the
canonical contract builder, copies its result, and replaces exactly:

- `unroll_length`: `12` to `24`;
- `total_steps`: `8_000_000` to `393_216`, the bounded evaluation budget.

The existing `tools/run_canonical_g1_shac.py` remains unchanged. A contract
test compares every produced key and fails if any field besides those two
differs. The dedicated parser exposes only solver profile, reference path,
seed, and output root; it exposes no resume or scientific override.

## Evaluation And Stop Gate

After finite completion, evaluate the final actor without replay from exact
reference phases `0`, `100`, `200`, `300`, and `400` under nominal physics.
Compare it with the horizon-12 step-393,216 vector
`14/43/47/61/19`, whose minimum is `14` and median is `43`.

Continue beyond this experiment only if all conditions hold:

1. Every required artifact and numeric training-state leaf is finite.
2. Worst-phase survival is at least `19` frames.
3. Median five-phase survival is at least `48` frames.
4. No evaluation uses replay, teleportation, or an intermediate reset as
   evidence of completion.

Any nonfinite update, missing artifact, minimum below `19`, or median below
`48` stops the fixed-24 branch. Passing the gate authorizes a separately
registered continuation; it does not establish full tracking, sim-to-real, or
hardware readiness. Full reference-suffix completion remains the eventual
success criterion.

## Verification

Use test-driven development for the contract builder and parser. Run the
focused runner tests, Ruff, Python compilation, exact diff checks, research
registry validation, and a guarded dry run before GPU execution. Execute the
registered experiment through `tools.runexp` and preserve all outputs under
its seed directory.
